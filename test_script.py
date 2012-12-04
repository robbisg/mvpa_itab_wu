path = '/media/DATA/fmri/learning'
conf = 'learning.conf'

subjects = os.listdir(path)
subjects = [s for s in subjects if s.find('_') == -1 and s.find('.') == -1]

masks = ['visual', 'll', 'ul', 'lr', 'ur', 'total']

analysis = ['spatial', 'spatiotemporal']

tasks = ['rest', 'task']

results = []

for mask in masks:
    res = analyze(path, subjects, spatiotemporal, tasks[1], conf, mask_area=mask)
    results.append(dict({'mask': mask,
                         'result': res
                         }))
    
##########################################################################