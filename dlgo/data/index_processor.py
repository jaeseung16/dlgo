import os
import sys
import multiprocessing
from urllib.request import urlopen, urlretrieve


def worker(url_and_target):
    try:
        (url, target_path) = url_and_target
        print('>>> Downloading ' + target_path)
        urlretrieve(url, target_path)
    except (KeyboardInterrupt, SystemExit):
        print('>>> Exiting child process')


class KGSIndex(object):
    def __init__(self, kgs_url='http://u-go.net/gamerecords/', index_page='kgs_index.html', data_directory='data'):
        """Create an index of zip files containing SGF data of actual Go Games on KGS.

        Parameters:
        -----------
        kgs_url: URL with links to zip files of games
        index_page: Name of local html file of kgs_url
        data_directory: name of directory relative to current path to store SGF data
        """
        self.kgs_url = kgs_url
        self.index_page = index_page
        self.data_directory = data_directory
        self.file_info = []
        self.urls = []
        self.load_index()
        print("KGSIndex initialized: # of file_info={}".format(len(self.file_info)))
        # print("file_info={}".format(self.file_info))

    def download_files(self):
        """Download zip files by distributing work on all available CPUs"""
        if not os.path.isdir(self.data_directory):
            os.makedirs(self.data_directory)

        urls_to_download = []
        for file_info in self.file_info:
            url = file_info['url']
            file_name = file_info['filename']
            if not os.path.isfile(self.data_directory + '/' + file_name):
                urls_to_download.append((url, self.data_directory + '/' + file_name))

        print("the number of urls to download={}".format(len(urls_to_download)))
        cores = multiprocessing.cpu_count()
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('fork')
        pool = multiprocessing.Pool(processes=cores)
        try:
            it = pool.imap(worker, urls_to_download)
            for _ in it:
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print(">>> Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def create_index_page(self):
        """If there is no local html containing links to files, create one."""
        if os.path.isfile(self.index_page):
            print('>>> Reading cached index page')
            index_file = open(self.index_page, 'r')
            index_contents = index_file.read()
            index_file.close()
        else:
            print('>>> Downloading index page')
            fp = urlopen(self.kgs_url)
            data = str(fp.read())
            fp.close()
            index_contents = data
            index_file = open(self.index_page, 'w')
            index_file.write(index_contents)
            index_file.close()
        return index_contents

    def load_index(self):
        """Create the actual index representation from the previously downloaded or cached html."""
        print("loading index")
        index_contents = self.create_index_page()
        split_page = [item for item in index_contents.split('<a href="') if item.startswith("https://")]
        for item in split_page:
            download_url = item.split('">Download')[0]
            if download_url.endswith('.tar.gz'):
                self.urls.append(download_url)
        for url in self.urls:
            filename = os.path.basename(url)
            split_file_name = filename.split('-')
            num_games = int(split_file_name[len(split_file_name) - 2])
            # print(filename + ' ' + str(num_games))
            self.file_info.append({'url': url, 'filename': filename, 'num_games': num_games})


if __name__ == '__main__':
    index = KGSIndex()
    index.download_files()
