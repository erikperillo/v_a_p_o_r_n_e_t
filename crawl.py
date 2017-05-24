from icrawler.builtin import BaiduImageCrawler
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import GoogleImageCrawler
import sys

keyword = "vaporwave art"
max_num = 6000

def baidu():
    crawler = BaiduImageCrawler(parser_threads=2, downloader_threads=4,
                                        storage={"root_dir": 'baidu'})
    crawler.crawl(keyword=keyword, max_num=max_num,
                         min_size=(200,200), max_size=None)

def bing():
    crawler = BingImageCrawler(parser_threads=2, downloader_threads=4,
                                        storage={"root_dir": 'bing'})
    crawler.crawl(keyword=keyword, max_num=max_num,
                         min_size=(200,200), max_size=None)

def google():
    crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                        storage={"root_dir": 'google'})
    crawler.crawl(keyword=keyword, max_num=max_num,
                         min_size=(200,200), max_size=None)

def main():
    if len(sys.argv) < 2:
        print("usage: {} <engine> (engine in {{baidu, bing, google}})".format(
            sys.argv[0]))
        return

    if sys.argv[1] == "baidu":
        baidu()
    elif sys.argv[1] == "bing":
        bing()
    elif sys.argv[1] == "google":
        google()
    else:
        print("not valid engine '{}'".format(sys.argv[1]))

if __name__ == "__main__":
    main()
