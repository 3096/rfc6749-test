#!/usr/bin/env python3
"""Generate and optionally rotate mock OIDC discovery content.

This script can either run once to produce a discovery document and JWKS, or run
continuously with asyncio so multiple periodic tasks can operate concurrently.

Examples:
    # Generate with a new RSA key
    python3 generate-mock-oidc-content.py \
        --base-url "https://<user>.github.io/oidc-scheduler-test" \
        --kid "test-key-v1" \
        --output-dir .

    # Rotate automatically and check GitHub Pages in parallel
    python3 generate-mock-oidc-content.py \
        --base-url "https://<user>.github.io/oidc-scheduler-test" \
        --output-dir . \
        --interval-seconds 60 \
        --pages-check-interval-seconds 15
"""

import argparse
import asyncio
import base64
import datetime
from dataclasses import dataclass
import json
import os
import shlex
import subprocess
import tempfile
import time
from typing import Optional
from urllib import parse as urllib_parse
import uuid


GO_STYLE_PAGES_CHECK_USER_AGENT = 'Go-http-client/2.0'


def log(message=''):
    """Print each output line with a UTC timestamp prefix."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    lines = str(message).splitlines()
    if not lines:
        lines = ['']
    for line in lines:
        print(f'[{timestamp}] {line}')


def int_to_base64url(n):
    """Convert a Python integer to base64url encoding (no padding)."""
    length = (n.bit_length() + 7) // 8
    raw = n.to_bytes(length, byteorder='big')
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def load_rsa_public_numbers_via_openssl(key_file):
    """
    Extract RSA public key modulus (n) and exponent (e) from a PEM private key
    using openssl CLI. Returns (n: int, e: int).
    """
    # Extract modulus
    result = subprocess.run(
        ['openssl', 'rsa', '-in', key_file, '-noout', '-modulus'],
        capture_output=True, text=True, check=True
    )
    # Output: "Modulus=ABCDEF..."
    mod_hex = result.stdout.strip().split('=', 1)[1]
    n = int(mod_hex, 16)

    # Extract public exponent
    result = subprocess.run(
        ['openssl', 'rsa', '-in', key_file, '-noout', '-text'],
        capture_output=True, text=True, check=True
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith('publicExponent:'):
            # "publicExponent: 65537 (0x10001)"
            e = int(line.split(':')[1].strip().split(' ')[0])
            return n, e

    raise RuntimeError("Could not extract public exponent from key file")


def generate_rsa_key(key_file):
    """Generate a 2048-bit RSA private key using openssl."""
    subprocess.run(
        ['openssl', 'genrsa', '-out', key_file, '2048'],
        capture_output=True, check=True
    )
    log(f"Generated RSA key: {key_file}")


@dataclass
class RuntimeState:
    """Track the most recent local, pushed, and remotely observed kids."""

    last_generated_kid: Optional[str] = None
    last_pushed_kid: Optional[str] = None
    last_seen_pages_kid: Optional[str] = None


async def run_command(args, capture_output=False):
    """Run a command asynchronously and raise on failure."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.PIPE if capture_output else None,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            args,
            output=stdout.decode() if stdout else None,
            stderr=stderr.decode() if stderr else None,
        )

    if not capture_output:
        return None

    return subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=stdout.decode() if stdout else '',
        stderr=stderr.decode() if stderr else '',
    )


def write_mock_oidc_content_sync(base_url, output_dir, kid, key_file):
    """Write the OpenID discovery document and JWKS for the provided key."""
    n, e = load_rsa_public_numbers_via_openssl(key_file)

    wellknown_dir = os.path.join(output_dir, '.well-known')
    os.makedirs(wellknown_dir, exist_ok=True)

    openid_config = {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        "userinfo_endpoint": f"{base_url}/userinfo",
        "jwks_uri": f"{base_url}/jwks.json",
    }
    openid_path = os.path.join(wellknown_dir, 'openid-configuration.json')
    with open(openid_path, 'w') as f:
        json.dump(openid_config, f, indent=2)
        f.write('\n')

    jwks = {
        "keys": [{
            "kty": "RSA",
            "kid": kid,
            "use": "sig",
            "alg": "RS256",
            "n": int_to_base64url(n),
            "e": int_to_base64url(e),
        }]
    }
    jwks_path = os.path.join(output_dir, 'jwks.json')
    with open(jwks_path, 'w') as f:
        json.dump(jwks, f, indent=2)
        f.write('\n')

    return openid_path, jwks_path


async def write_mock_oidc_content(base_url, output_dir, kid, key_file):
    """Write mock OIDC content without blocking the event loop."""
    return await asyncio.to_thread(
        write_mock_oidc_content_sync,
        base_url,
        output_dir,
        kid,
        key_file,
    )


async def run_git_command(repo_dir, args, capture_output=False):
    """Run a git command in the given repo directory."""
    return await run_command(['git', '-C', repo_dir] + args, capture_output=capture_output)


async def ensure_git_repo(repo_dir):
    """Fail early if the output directory is not inside a git repository."""
    try:
        result = await run_git_command(repo_dir, ['rev-parse', '--show-toplevel'], capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Output directory {repo_dir} is not inside a git repository"
        ) from exc

    return result.stdout.strip()


async def ensure_git_remote(repo_dir, remote_name):
    """Fail early if the requested git remote does not exist."""
    try:
        await run_git_command(repo_dir, ['remote', 'get-url', remote_name], capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Git remote '{remote_name}' is not configured for {repo_dir}"
        ) from exc


async def has_staged_changes(repo_dir, staged_paths):
    """Return True when the requested paths have staged changes."""
    process = await asyncio.create_subprocess_exec(
        'git', '-C', repo_dir, 'diff', '--cached', '--quiet', '--', *staged_paths,
    )
    returncode = await process.wait()
    if returncode == 0:
        return False
    if returncode == 1:
        return True
    raise RuntimeError('Failed to inspect staged git changes')


async def commit_and_push(repo_dir, remote_name, branch_name, commit_message, staged_paths):
    """Stage generated artifacts, commit them, and push to the remote branch."""
    await run_git_command(repo_dir, ['add', '--'] + staged_paths)

    if not await has_staged_changes(repo_dir, staged_paths):
        log('No git changes detected after staging; skipping commit and push')
        return False

    await run_git_command(repo_dir, ['commit', '--only', '-m', commit_message, '--'] + staged_paths)
    await run_git_command(repo_dir, ['push', remote_name, branch_name])
    return True


def generate_rotation_kid(kid_prefix):
    """Generate a unique kid for a rotation event."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = uuid.uuid4().hex[:8]
    return f"{kid_prefix}-{timestamp}-{suffix}"


def build_rotation_key_path(state_dir, kid):
    """Build a stable path for each generated rotation key."""
    return os.path.join(state_dir, f'{kid}.pem')


def print_summary(base_url, output_dir, kid, key_file, openid_path, jwks_path):
    """Print a consistent summary after content generation."""
    log(f"Generated mock OIDC content in {output_dir}")
    log(f"OpenID Config: {base_url}/.well-known/openid-configuration.json")
    log(f"JWKS:          {base_url}/jwks.json")
    log(f"Key ID (kid):  {kid}")
    log(f"Key file:      {key_file}")
    log('To verify locally:')
    log(f"cat {shlex.quote(openid_path)}")
    log(f"cat {shlex.quote(jwks_path)}")


async def fetch_json(url, timeout_seconds):
    """Fetch JSON using a Go-like curl request"""
    result = await run_command([
        'curl',
        '--silent',
        '--show-error',
        '--fail',
        '--location',
        '--http2',
        '--tlsv1.2',
        '--max-time',
        str(timeout_seconds),
        '--user-agent',
        GO_STYLE_PAGES_CHECK_USER_AGENT,
        url,
    ], capture_output=True)
    return json.loads(result.stdout)


def add_cache_buster(url):
    """Append a cache-busting query parameter to a URL."""
    parsed = urllib_parse.urlsplit(url)
    query_items = urllib_parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_items.append(('ts', str(int(time.time()))))
    new_query = urllib_parse.urlencode(query_items)
    return urllib_parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))


def extract_first_kid(jwks):
    """Return the first kid from a JWKS payload if present."""
    keys = jwks.get('keys', [])
    if not keys:
        return None
    return keys[0].get('kid')


async def perform_rotation(base_url, output_dir, kid_prefix, repo_dir, remote_name, branch_name, state_dir, state):
    """Generate a new key, update artifacts, commit, and push them."""
    kid = generate_rotation_kid(kid_prefix)
    key_file = build_rotation_key_path(state_dir, kid)

    log(f"[rotation] Generating rotated key {kid}")
    await asyncio.to_thread(generate_rsa_key, key_file)
    openid_path, jwks_path = await write_mock_oidc_content(base_url, output_dir, kid, key_file)
    print_summary(base_url, output_dir, kid, key_file, openid_path, jwks_path)

    staged_paths = [
        os.path.relpath(openid_path, repo_dir),
        os.path.relpath(jwks_path, repo_dir),
    ]
    commit_message = f'Rotate mock OIDC key: kid={kid}'
    pushed = await commit_and_push(repo_dir, remote_name, branch_name, commit_message, staged_paths)
    state.last_generated_kid = kid
    if pushed:
        state.last_pushed_kid = kid
        log(f'Pushed commit: {commit_message}')


async def check_pages_status(pages_check_url, pages_check_timeout_seconds, state):
    """Poll the deployed JWKS and report whether Pages has caught up."""
    checked_url = add_cache_buster(pages_check_url)
    try:
        payload = await fetch_json(checked_url, pages_check_timeout_seconds)
    except (subprocess.CalledProcessError, TimeoutError, json.JSONDecodeError) as exc:
        log(f'[pages] Check failed for {pages_check_url}: {exc}')
        return

    deployed_kid = extract_first_kid(payload)
    if deployed_kid is None:
        log(f'[pages] No kid found in {pages_check_url}')
        return

    if deployed_kid != state.last_seen_pages_kid:
        log(f'[pages] Deployed kid is now {deployed_kid}')
        state.last_seen_pages_kid = deployed_kid

    if state.last_pushed_kid:
        if deployed_kid == state.last_pushed_kid:
            log(f'[pages] GitHub Pages is serving the latest pushed kid {deployed_kid}')
        else:
            log(
                '[pages] Waiting for Pages to catch up: '
                f'deployed={deployed_kid}, latest-pushed={state.last_pushed_kid}'
            )


async def run_periodic_task(name, interval_seconds, action, stop_event):
    """Run an action repeatedly on its own interval until cancelled."""
    while not stop_event.is_set():
        started_at = time.monotonic()
        try:
            await action()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log(f'[{name}] Task failed: {exc}')

        remaining = interval_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            pass


async def run_service_mode(
    base_url,
    output_dir,
    kid_prefix,
    interval_seconds,
    remote_name,
    branch_name,
    pages_check_interval_seconds,
    pages_check_url,
    pages_check_timeout_seconds,
):
    """Run concurrent periodic tasks for rotation and deployment checks."""
    state = RuntimeState()
    stop_event = asyncio.Event()
    tasks = []

    if interval_seconds is not None:
        repo_dir = await ensure_git_repo(output_dir)
        await ensure_git_remote(repo_dir, remote_name)
        state_dir = os.path.join(output_dir, '.rotation-state')
        os.makedirs(state_dir, exist_ok=True)

        log(f'Starting automatic rotation every {interval_seconds} seconds')
        log(f'Repo:          {repo_dir}')
        log(f'Remote/branch: {remote_name}/{branch_name}')

        tasks.append(asyncio.create_task(run_periodic_task(
            'rotation',
            interval_seconds,
            lambda: perform_rotation(
                base_url,
                output_dir,
                kid_prefix,
                repo_dir,
                remote_name,
                branch_name,
                state_dir,
                state,
            ),
            stop_event,
        )))

    if pages_check_interval_seconds is not None:
        effective_pages_check_url = pages_check_url or f'{base_url}/jwks.json'
        log(f'Starting Pages check every {pages_check_interval_seconds} seconds')
        log(f'Check URL:      {effective_pages_check_url}')
        tasks.append(asyncio.create_task(run_periodic_task(
            'pages',
            pages_check_interval_seconds,
            lambda: check_pages_status(
                effective_pages_check_url,
                pages_check_timeout_seconds,
                state,
            ),
            stop_event,
        )))

    if not tasks:
        raise RuntimeError('No periodic tasks configured')

    log('Stop with Ctrl-C')

    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log('Service stopped')


async def main_async():
    parser = argparse.ArgumentParser(
        description='Generate mock OIDC discovery endpoint content for manual testing.'
    )
    parser.add_argument(
        '--base-url', required=True,
        help='Base URL of the static site (e.g., https://oidcmocktest.z5.web.core.windows.net)'
    )
    parser.add_argument(
        '--key-file',
        help='Path to RSA private key PEM file. If not provided or file does not exist, a new key is generated.'
    )
    parser.add_argument(
        '--kid', default='test-key-v1',
        help='Key ID (kid) for the JWK (default: test-key-v1)'
    )
    parser.add_argument(
        '--kid-prefix', default='test-key',
        help='Prefix used when generating a new kid on each rotation (default: test-key)'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory for the generated content'
    )
    parser.add_argument(
        '--interval-seconds', type=int,
        help='If set, continuously rotate keys at this interval and push each change'
    )
    parser.add_argument(
        '--remote', default='origin',
        help='Git remote to push to in rotation mode (default: origin)'
    )
    parser.add_argument(
        '--branch', default='main',
        help='Git branch to push to in rotation mode (default: main)'
    )
    parser.add_argument(
        '--pages-check-interval-seconds', type=int,
        help='If set, poll the deployed JWKS on a separate interval'
    )
    parser.add_argument(
        '--pages-check-url',
        help='JWKS URL to poll in Pages check mode (default: <base-url>/jwks.json)'
    )
    parser.add_argument(
        '--pages-check-timeout-seconds', type=int, default=10,
        help='Timeout for each Pages check request in seconds (default: 10)'
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip('/')
    output_dir = os.path.abspath(args.output_dir)

    if args.interval_seconds is not None and args.interval_seconds <= 0:
        raise SystemExit('--interval-seconds must be greater than 0')
    if args.pages_check_interval_seconds is not None and args.pages_check_interval_seconds <= 0:
        raise SystemExit('--pages-check-interval-seconds must be greater than 0')
    if args.pages_check_timeout_seconds <= 0:
        raise SystemExit('--pages-check-timeout-seconds must be greater than 0')

    if args.interval_seconds is not None or args.pages_check_interval_seconds is not None:
        await run_service_mode(
            base_url=base_url,
            output_dir=output_dir,
            kid_prefix=args.kid_prefix,
            interval_seconds=args.interval_seconds,
            remote_name=args.remote,
            branch_name=args.branch,
            pages_check_interval_seconds=args.pages_check_interval_seconds,
            pages_check_url=args.pages_check_url,
            pages_check_timeout_seconds=args.pages_check_timeout_seconds,
        )
        return

    key_file = args.key_file
    if not key_file:
        key_file = os.path.join(tempfile.gettempdir(), 'mock-rsa-key.pem')

    if not os.path.exists(key_file):
        await asyncio.to_thread(generate_rsa_key, key_file)

    openid_path, jwks_path = await write_mock_oidc_content(base_url, output_dir, args.kid, key_file)
    print_summary(base_url, output_dir, args.kid, key_file, openid_path, jwks_path)


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
