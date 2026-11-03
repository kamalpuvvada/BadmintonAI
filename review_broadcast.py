"""Open a local broadcast review with seekable video; no uploads or writes."""
import argparse
from pathlib import Path
import threading
import webbrowser

from academy_app import LocalResults


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).parent/'out/axelsen-lee-analysis-008')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    if not (args.directory/'index.html').is_file():parser.error('Review page not found in this folder')
    server=LocalResults()
    try:
        url=server.url(args.directory)
        print(url,flush=True)
        if not args.no_browser:webbrowser.open(url)
        print('Review is open locally. Close this window when finished.',flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:pass
    finally:server.close()


if __name__=='__main__':main()
