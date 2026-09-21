import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory

from static_server import make_server


class NoCacheServerTest(unittest.TestCase):
    def test_html_and_modules_are_not_cached(self):
        with TemporaryDirectory() as directory:
            Path(directory, 'index.html').write_text('<title>test</title>')
            Path(directory, 'app.mjs').write_text('export default 1;')
            server = make_server('127.0.0.1', 0, directory)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for path in ('/', '/app.mjs'):
                    connection = HTTPConnection('127.0.0.1', server.server_port)
                    connection.request('HEAD', path)
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.getheader('Cache-Control'), 'no-store, max-age=0')
                    connection.close()
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
