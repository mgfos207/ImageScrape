'''
Author Micah Forster
Date 04/22/2022
'''
import logging
import os
import json
import urllib.request
import argparse
import pathlib
import threading
import time
import math

from serpapi import GoogleSearch

class ImageDownloadThread(threading.Thread):
    def __init__(self, thread_id, name,logger, img_list, save_results_path, folder_name):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.save_results_path = save_results_path
        self.folder_name = folder_name
        self.img_list = img_list
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.logger = logger

    def persist_images(self):
        for idx,image in enumerate(self.img_list):
            self.logger.info("Saving file {}".format(image))
            opener=urllib.request.build_opener()
            opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582')]
            urllib.request.install_opener(opener)
            image_name = image['original']
            #TO-DO find a better way to get image suffix
            suffix = pathlib.Path(image_name).suffix
            try:
                urllib.request.urlretrieve(image['original'], f'{self.save_results_path}/{self.folder_name}_{self.thread_id}_{idx}.jpg')
            except Exception as e:
                self.logger.error(f"Issue downloading image {e}")
        self.logger.info(f"Ended job for fetching images for thread: {self.thread_id}")
        
    def run(self):
        try:
            self.start_time = time.time()
            self.logger.info(f"Starting the save image process for thread {self.thread_id}")
            self.persist_images()
        except Exception as e:
            print(f"Having problems persisting the images from the data onto file system for thread id: {self.thread_id} with error: {e}")
        finally:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            print(f"Successfully completed saving images for thread {self.thread_id}. Completing in {self.duration} seconds")

#This method allows for the user to fetch images off the internet
#Using the SeraAPI to do a google search sans having to do precocessing
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [QUERY]...",
        description="Run Google Search Query for image results and save results to file system."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('--query', type=str, required=True)
    
    return parser

def fetchImages(config, num=1, query=None) -> list:
    #fetch the data if query is set
    image_results_collection = list()
    if query is not None:
        increment = 1
        
        #make repeated requests to Google API until all the requested info is complete
        while increment <= num:
            params = {
                'api_key': os.getenv('GS_APIKEY'),
                'engine': config['search_engine'],
                'tbm': config['tbm'],
                'q': query
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            
            #Ensure that there are results in the query otherwise break out of while loop
            image_results = results['images_results']
            if len(image_results) > 0:
                image_results_collection = image_results + image_results_collection
            else:
                break
            increment += 1
    
    return image_results_collection

def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    query = args.query
    folder_name = query.replace(" ", "_")
    logging.basicConfig(filename=f'{folder_name}.log', encoding='utf-8', level=logging.DEBUG)
    logging.info(f"Query being passed: {query}")
    logging.info("Starting job for fetching images")
    #Set the global config
    path = os.getcwd()
    with open("{}/config/image_scrapper_config.json".format(path)) as config_file:
        config = json.load(config_file)


    # try:
    #     #Get image data using the threading class
    #     imgs_data = get_img_data(df)
    #     #split the data into chunks for the 10 threads
    #     data_length = len(imgs_data)
    #     chunks = math.ceil(data_length / 50)
    #     #store the chunks into different arrays perhaps a list of lists
    #     chunked_list = [imgs_data[i: i+chunks] for i in range(0, data_length, chunks)]
    #     #create different threads
    #     for idx, chunk in enumerate(chunked_list):
    #         img_thread = ImageDownloadThread(idx, f'img_thread-{idx}', chunk)
    #         img_thread.start()
    # except Exception as e:
    #     print(f"Having problems persisting the images from the data onto file system: {e}")

    try:
        #get the images from search query
        image_results = fetchImages(config['serapi'], 1, query)
    except Exception as e:
        logging.error(f"Got the following error from running the query: {e}")

    if len(image_results) > 0:

        #save images to file
        save_results_path = "{0}/{1}".format(path, folder_name)
        if not os.path.exists(save_results_path):
            os.makedirs(save_results_path)

        #save image results into a file (jic the next part of saving the acutal fails, and then it can be done manually)
        with open(f'{save_results_path}/{folder_name}_results.json', 'w') as result_file:
            json.dump(image_results, result_file)

        
        #Create the threads for fetching the images
        num_threads = config['num_threads']
        data_length = len(image_results)
        chunks = math.ceil(data_length/num_threads)
        chunked_list = [image_results[i: i+chunks] for i in range(0, data_length, chunks)]

        logging.info("Starting the threads for fetching images from Google Serapi")
        try:
            for idx, chunk in enumerate(chunked_list):
                img_thread = ImageDownloadThread(idx, f'img_thread-{idx}', logging, chunk, save_results_path, folder_name)
                img_thread.start()
        except Exception as e:
            logging.error(f"Error fetching all the images due to the following error: {e}")
        finally:
            logging.info("Finished process of fetching images based on user query")
        # for index, image in enumerate(image_results):
        #     logging.info("Saving file {}".format(image))
        #     opener=urllib.request.build_opener()
        #     opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582')]
        #     urllib.request.install_opener(opener)
        #     image_name = image['original']
        #     #TO-DO find a better way to get image suffix
        #     suffix = pathlib.Path(image_name).suffix
        #     try:
        #         urllib.request.urlretrieve(image['original'], f'{save_results_path}/{folder_name}_{index}.jpg')
        #     except Exception as e:
        #         logging.error(f"Issue downloading image {e}")

        # logging.info("Ended job for fetching images")


if __name__ == '__main__':
    main()