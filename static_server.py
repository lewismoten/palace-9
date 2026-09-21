#!/usr/bin/env python3
"""Loopback static server for Noughtlear Proliferation.

Every response is explicitly non-cacheable so Tailnet clients receive the
current browser modules after a local update.
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CACHE_CONTROL = 'no-store, max-age=0'


class NoCacheStaticHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header('Cache-Control', CACHE_CONTROL)
        self.send_header('Pragma', 'no-cache')
        super().end_headers()


def make_server(host: str, port: int, directory: str | Path) -> ThreadingHTTPServer:
    handler = partial(NoCacheStaticHandler, directory=str(Path(directory).resolve()))
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve static project files without client caching.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=18777, type=int)
    parser.add_argument('--directory', default=Path(__file__).parent)
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.directory)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
