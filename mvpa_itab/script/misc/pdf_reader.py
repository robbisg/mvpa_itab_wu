import os
import numpy as np
path_ = '/home/robbis/rossep/'
dirs_ = os.listdir(path_)

dirs_.sort()
dirs_ = dirs_[:-1]

selected = []
total_dirs = len(dirs_)
for i,d in enumerate(dirs_):
    path_file = os.path.join(path_, d)
    
    output = os.popen('ls '+path_file+'/*.txt').read()
    outlist = output.split('\n')
    outlist.sort()
    
    total_files = len(outlist[1:])
    for j,f in enumerate(outlist[1:]):
        
        cat_ = os.popen('cat '+f).read()
        
        conditions = np.array([cat_.find('figure(') != -1,
                      cat_.find('<?xml') == -1,
                      cat_.find('import') != -1,
                      cat_.find('Package:') == -1
                      ], dtype=np.bool)
        
        if np.all(conditions):
            selected.append(f)
        
        progress = 100. * ((i*total_files)+(j+1))/(total_dirs * total_files)
        update_progress(progress)



command = 'curl -LH "Accept: text/bibliography; style=bibtex" http://dx.doi.org/10.1101/086637'





def pdf2txt(filepath):
    import os, subprocess
    args = ["/usr/bin/pdftotext",
            '-enc',
            'UTF-8',
            "%s" %(filepath),
            '-']
    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = res.stdout.decode('utf-8')

    return output


def getInfo(filename):
    
    try:
        pdf_file = PyPDF2.PdfFileReader(open(filename, 'rb'))
        info = pdf_file.getDocumentInfo()
    except Exception as e:
        return {}
    
    if info is None:
        return {}

    return info

def requests_by_doi(doi):
    import requests
    url = "http://dx.doi.org/%s" % (doi)
    print(url)
    try:
        response = requests.get(url, headers={'style':'bibtex', "Accept":"text/bibliography"})
    except Exception as err:
        try:
            response = requests.get(url, headers={'style':'bibtex', "Accept":"text/bibliography"})
        except Exception as err:
            print(err)
            return {}
    
    response = response.content.decode()

    if len(response) == 0:
        return {}

    if response[0] == "<":
        print("doi html: "+doi)
        return {}
    else:
        print(response)

    output = {}
    # Author
    try:
        author = response.split(",")[0]
    except expression as identifier:
        author = ''
    
    # Year
    try:
        year = response.split("(")[1][:4]
    except Exception as identifier:
        year = ''
    
    # Journal
    year_position = response.find("(")
    try:
        #journal = response.split("(")[1].split(".")[-1].split(",")[0]
        journal = response[year_position:].split(".")[2].split(",")[0]
        if journal[1].isnumeric():
            journal = 'BioarXiv'
        else:
            journal = journal.replace(" ", "")
    except Exception as err:
        journal = ''

    output['Author'] = author
    output['Year'] = year
    output['Journal'] = journal
    output['FullEntry'] = response
    print(output)
    return output


def parse_field(f, info):
    output = dict()
    keywords = ["/Author", "/Title", "/doi", "/Keywords", "/Subject"]
    print(f)

    for key in keywords:

        data = info.get(key)
        if data == None:
            continue

        if key == "/doi":
            print("doi in info", f, data)
            doi_ = data.split("/")
            if "." not in doi_[1]:
                return output
            
            doi_info = doi_[1].split(".") 
            if len(doi_info) > 2:
                output['Journal'] = doi_info[1]
                output['Year'] = doi_info[2] 
            else:
                output['Journal'] = doi_info[1]    
        
        elif key == "/Author":
            #print([key, data])
            if isinstance(data, str) and data is not "":
                output['Author'] = data.split(" ")[-1]

        elif key == "/Title":
            if isinstance(data, str) and len(data.split(" ")) > 5:
                title_items = data.split(" ")[:4]
                output['Title'] = "_".join(title_items)
            elif isinstance(data, str) and data.find("doi") != -1:
                key = "/Subject"
        
        elif key == "/Subject":
            print(f, data)  
            try:
                has_doi = data.find("doi:") != -1
            except Exception as e:
                continue

            if has_doi:
                doi = data.split("doi:")[-1]
                output_ = requests_by_doi(doi)
                output.update(output_)
                print(output)

    return output

def doi_splitter(doi, delimiter):
    doi = doi.split(delimiter)
    if len(doi) > 2:
        if len(doi[0]) == 0:
            doi = doi[1]
        else:
            doi = doi[0]
    else:
        if len(doi[0]) == 0:
            doi = doi[1]
        else:
            doi = doi[0]

    return doi





def pdf_scraper(filename, info):
    output = pdf2txt(filename)
    parsed_info = {}
    print(filename, info)
    for doi_string in ['DOI:', "doi:", "http://dx.doi.org/", "doi/", "DOI "]:
        if output.find(doi_string) != -1:

            doi_position = output.find(doi_string)+len(doi_string)
            doish = output[doi_position:doi_position+50]
            doi = doish.split("\n")[0]
            if len(doi) == 0:
                doi = doish.split("\n")[1]
            
            for delimiter in [' ', ',', ')', '\t', 'http:', '\n']:
                doi = doi_splitter(doi, delimiter)

            if doi[-1] == '.':
                doi = doi[:-1]

            print("parsed doish", filename, repr(doish), repr(doi))
            parsed_info = requests_by_doi(doi)
        
        if parsed_info != {}:
            break

    return parsed_info

def frontiers_parser(filename, info):
    fname = filename.split("/")[-1]
    output = parse_field(filename, info)
    
    output['Year'] = info['/ModDate'][2:6]
    output['Journal'] = fname.split("-")[0]
    return output

def elife_parser(filename, info):
    id_ = filename.split("/")[-1].split("-")[1]
    doi = "http://dx.doi.org/10.7554/eLife.%s" % (id_)
    output = requests_by_doi(doi)
    output['Journal'] = 'elife'
    return output


def bioarxiv_parser(filename):
    id_ = filename.split("/")[-1].split(".")[0]
    doi = "10.1101/%s" % (id_)
    return requests_by_doi(doi)


def arxiv_parser(filename):
    from arxiv2bib import arxiv2bib
    id_ = filename.split("/")[-1].split(".pdf")[0]
    ref = arxiv2bib([id_])[0]
    output = {}
    output['Author'] = ref.authors[-1].split(" ")[-1]
    output['Year'] = ref.years
    output['Journal'] = 'arXiv'

    return output

def plos_parser(filename):
    id_ = filename.split("/")[-1].split(".pdf")[0]
    doi = "10.1371/%s" % (id_)
    return requests_by_doi(doi)


def cerebcortex_parser(f, info):
    id_ = info["/Title"].split(" ")[0]
    doi = "10.1093/cercor/%s" % (id_)
    return requests_by_doi(doi)


def scihub_parser(f):
    id_ = f.split("/")[-1].split(".pdf")[0]
    id_ = id_.replace("@", "/")
    return requests_by_doi(id_)


def nature_parser(f):
    id_ = f.split("/")[-1].split(".pdf")[0]
    doi = "10.1038/%s" % (id_)
    return requests_by_doi(doi)


import glob
import PyPDF2
from tqdm import tqdm
path = "/home/robbis/MEGAsync/renamed"
path = "/home/robbis/MEGAsync/PAPER/MVPA"
path = '/home/robbis/CloudMEGA/'
pdf_list = glob.glob("%s/*.pdf" % (path))
pdf_list.sort()

db = dict()
rejected = dict()
for f in pdf_list:
    print("----")
    info = getInfo(f)
    if info == None:
        continue
    
    fname = f.split("/")[-1]
    """
    if f.find(".full.") != -1:
        parsed_info = bioarxiv_parser(f)
    elif f.find('/fn') != -1 or f.find('/fp') != -1:
        parsed_info = frontiers_parser(f, info)
    elif f.find("elife") != -1:
        parsed_info = elife_parser(f, info)
    elif fname[:4].isnumeric() and fname[4] == '.' and fname[5].isnumeric():
        parsed_info = arxiv_parser(f)
    elif f.find("Cereb.") != -1:
        parsed_info = cerebcortex_parser(f, info)
    elif f.find("journal.") != -1:
        parsed_info = plos_parser(f)
    elif '/doi' in info.keys():
        parsed_info = requests_by_doi(info['/doi'])
    elif '/Title' in info.keys() and info['/Title'].find("doi:") != -1:
        parsed_info = requests_by_doi(info['/Title'])
    elif f.find("@") != -1:
        parsed_info = scihub_parser(f)
    elif fname[0] == 'n':
        parsed_info = nature_parser(f)
    else:
    """
    parsed_info = pdf_scraper(f, info)
    list_values = list(parsed_info.values())

    if any(x == None for x in list_values) or any(x == '' for x in list_values):
        rejected[f] = info
    elif parsed_info == {}:
        rejected[f] = info
    else:
        db[f] = parsed_info

count_dictionary = {}
titles_dictionary = {}
for f, items in db.items():

    try:
        key = "%s_%s_%s" % (items['Author'].replace(" ",""), items['Year'], items['Journal'])
        if key in count_dictionary.keys():
            count_dictionary[key] += 1
            key += "_"+str(count_dictionary[key])
        else:
            count_dictionary[key] = 0
        new_filename = "%s.pdf" % (key)
        command = "mv -n %s %s" % (f, os.path.join(path, 'renamed', new_filename))
        print(command)

    except Exception:
        continue


path = "/home/robbis/Copy/MEGAsync/SyncDebris/2019-02-26/"
pdf_list = glob.glob("%s/*.pdf" % (path))