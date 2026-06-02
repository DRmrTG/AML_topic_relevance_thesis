import json
import os
import time
import requests


api_key = "REDACTED"
base_url = "https://content.guardianapis.com/search"
query = "launder OR laundered OR laundering OR launders OR launderers OR launderer"
from_date = "1999-01-01"
to_date = "2026-04-22"
page_size = 200
#Keeping track of daily limit
daily_limit = 500

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(script_dir, "raw_data")
output_json = os.path.join(raw_data_dir, "guardian_launder_1999_2026.json")

#the checkpoint file to save progress in case of time-out
checkpoint_file = os.path.join(raw_data_dir, ".checkpoint.json")


#Verifying checkpoint and loading if this is the case
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        #open and load the checkpoint file
        f = open(checkpoint_file, "r", encoding="utf-8")
        cp = json.load(f)
        f.close()
        number_already_downloaded = len(cp["results"])
        next_page_number = cp["next_page"]
        print("found checkpoint! already downloaded " + str(number_already_downloaded) + " articles")
        print("resuming from page " + str(next_page_number))
        return cp["results"], cp["next_page"]
    #if no checkpoint file exists, start fresh
    print("no checkpoint found - starting from the beginning")
    return [], 1

#Saving checkpoint
def save_checkpoint(results, next_page):
    #build a dictionary with the current results and the next page to retrieve
    checkpoint_data = {
        "results": results,
        "next_page": next_page
    }
    #save it to the checkpoint file
    f = open(checkpoint_file, "w", encoding="utf-8")
    json.dump(checkpoint_data, f)
    f.close()


def save_output(results):
    rows = []
    for article in results:
        fields = article.get("fields", {})
        all_tags = article.get("tags", [])
        tag_titles = []
        for t in all_tags:
            tag_title = t.get("webTitle", "")
            tag_titles.append(tag_title)
        keywords = " | ".join(tag_titles)
        
        #build the cleaned up article dictionary
        cleaned_article = {
            "id": article.get("id"),
            "date": article.get("webPublicationDate"),
            "section": article.get("sectionName"),
            "url": article.get("webUrl"),
            "title": article.get("webTitle"),
            "trailText": fields.get("trailText"),
            "byline": fields.get("byline"),
            "bodyText": fields.get("bodyText"),
            "tags_keywords": keywords,
        }
        rows.append(cleaned_article)
    
    #save all the cleaned articles to the output json file
    f = open(output_json, "w", encoding="utf-8")
    json.dump(rows, f, indent=4, ensure_ascii=False)
    f.close()
    print("saved " + str(len(rows)) + " articles to " + output_json)


#this function deletes the checkpoint file once everything is downloaded
#i don't need it anymore once the download is complete
def delete_checkpoint():
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("checkpoint file deleted - download is complete!")


#this function makes a single API call to the guardian
#it returns the parsed json response or crashes if something went wrong
def guardian_get(params):
    #make the request with a 30 second timeout
    #the timeout means it won't hang forever if the server doesn't respond
    response = requests.get(base_url, params=params, timeout=30)
    
    #check if the request was successful
    #status code 200 means everything is ok
    #anything else means something went wrong
    if response.status_code != 200:
        #print the error and stop
        error_message = "HTTP error " + str(response.status_code)
        raise RuntimeError(error_message)
    
    #parse and return the json response
    return response.json()


#downloading function for all the articles
def fetch():
 
    os.makedirs(raw_data_dir, exist_ok=True)
    results, start_page = load_checkpoint()
    #tracking calls used
    calls_used = 0
    
    #API parameters for request
    base_params = {
        "api-key": api_key,
        "q": query,
        "from-date": from_date,
        "to-date": to_date,
        "page-size": page_size,
        "show-fields": "all",  
        "show-tags": "keyword"
    }
    
    #first i need to find out how many pages there are in total
    #i do this by fetching the first page and looking at the response
    #i do this even on a resume in case the total number of pages changed
    probe_params = {
        "api-key": api_key,
        "q": query,
        "from-date": from_date,
        "to-date": to_date,
        "page-size": page_size,
        "show-fields": "all",
        "show-tags": "keyword",
        "page": 1,
    }
    
    #make the first request to find out how many pages there are
    probe = guardian_get(probe_params)
    calls_used = calls_used + 1
    
    #extract the total number of pages and articles from the response
    response_data = probe.get("response", {})
    total_pages = response_data.get("pages", 1)
    total_results = response_data.get("total", "?")
    
    print("searching for: " + query)
    print("date range: " + from_date + " to " + to_date)
    print("total articles found by api: " + str(total_results))
    print("total pages to download: " + str(total_pages))
    
    #if we're starting fresh (not resuming) collect the results from page 1
    #we already have page 1 data from the probe request so no need to fetch it again
    if start_page == 1:
        page_1_results = response_data.get("results", [])
        for r in page_1_results:
            results.append(r)
        #save progress and move on to page 2
        save_checkpoint(results, 2)
        print("page 1 of " + str(total_pages) + " - " + str(len(results)) + " articles so far")
        #wait 1 second between requests to be polite to the api
        time.sleep(1)
        start_page = 2
    
    #now download the remaining pages one by one and anticipate errors
    try:
        for page in range(start_page, total_pages + 1):
            if calls_used >= daily_limit:
                print("reached daily limit of " + str(daily_limit) + " api calls!")
                print("downloaded " + str(len(results)) + " articles so far")
                print("will resume from page " + str(page) + " tomorrow")
                #save progress 
                save_checkpoint(results, page)
                #save whatever downloaded so far
                save_output(results)
                return
            
            #build the params for this page
            params = {}
            for key in base_params:
                params[key] = base_params[key]
            params["page"] = page
            #fetch this page
            page_data = guardian_get(params)
            calls_used = calls_used + 1
            #extract the results from this page and add them to the main list
            page_results = page_data.get("response", {}).get("results", [])
            for r in page_results:
                results.append(r)
            #print progress so i know it's still running
            print("page " + str(page) + " of " + str(total_pages) + " - " + str(len(results)) + " articles so far - call " + str(calls_used) + " of " + str(daily_limit))
            #save checkpoint after every page so i don't lose progress
            save_checkpoint(results, page + 1)
            #wait 1 second between requests to avoid getting blocked by the api
            time.sleep(1)
    
    #if the user presses ctrl+c to stop the script, save progress before exiting
    except KeyboardInterrupt:
        print("script interrupted by user!")
        print("saving checkpoint before exiting...")
        save_checkpoint(results, page)
        save_output(results)
        print("safe to exit - run the script again to continue from where i left off")
        return
    
    #if something else goes wrong, save progress before crashing
    except Exception as e:
        print("something went wrong: " + str(e))
        print("saving checkpoint before exiting...")
        save_checkpoint(results, page)
        save_output(results)
        #re-raise the error so i can see what went wrong
        raise
    
    #if we get here everything downloaded successfully
    save_output(results)
    delete_checkpoint()
    print("all done! " + str(len(results)) + " articles saved to " + output_json)


#this is the entry point of the script
#it only runs fetch() if i run this file directly
#if i import this file from another script, fetch() won't run automatically
if __name__ == "__main__":
    fetch()