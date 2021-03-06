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
from serpapi import GoogleSearch

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

    #get the images from search query
    image_results = fetchImages(config, 1, query)

    #save images to file
    save_results_path = "{0}/{1}".format(path, folder_name)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    #save image results into a file (jic the next part of saving the acutal fails, and then it can be done manually)
    with open(f'{save_results_path}/{folder_name}_results.json', 'w') as result_file:
        json.dump(image_results, result_file)

    
    for index, image in enumerate(image_results):
        logging.info("Saving file {}".format(image))
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582')]
        urllib.request.install_opener(opener)
        image_name = image['original']
        #TO-DO find a better way to get image suffix
        suffix = pathlib.Path(image_name).suffix
        try:
            urllib.request.urlretrieve(image['original'], f'{save_results_path}/{folder_name}_{index}.jpg')
        except Exception as e:
            logging.error(f"Issue downloading image {e}")

    logging.info("Ended job for fetching images")


if __name__ == '__main__':
    main()