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


def update_progress(progress):
    sys.stdout.write( '\r[{0}] {1}%'.format('#'*np.int(progress/10), progress))
    sys.stdout.flush()  






import os, subprocess

def pdf2txt(filepath):

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
    
    return info



def parse_field(key, data):
    output = dict()
    if data == None:
        output[key] = None
        return output

    if key == "/doi":
        print(data)
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

    return output    






def process_info(info):


    keywords = ["/Author", "/Title", "/doi", "/Keywords", "/Subject"]
    item = dict()

    for key in keywords:
        item.update(parse_field(key, info.get(key)))
    return item




import glob
import PyPDF2
path = "/home/robbis/Downloads/"
pdf_list = glob.glob("%s/*.pdf" % (path))

db = dict()
rejected = dict()
for f in pdf_list:
    
    info = getInfo(f)
    if info == None:
        continue
    parsed_info = process_info(info)
    list_values = list(parsed_info.values())

    if any(x == None for x in list_values) or any(x == '' for x in list_values):
        rejected[f] = info
    else:
        db[f] = parsed_info


for f, items in db.items():

    if "Year" not in items.keys():
        continue
    new_filename = "%s_%s_%s.pdf" % (items['Author'], items['Year'], items['Journal'])
    command = "mv %s %s" % (f, os.path.join(path, new_filename))
    print(command)