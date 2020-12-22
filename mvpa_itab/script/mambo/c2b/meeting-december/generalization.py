from joblib import load
import glob
folders = glob.glob("/media/robbis/DATA/meg/c2b/meeting-december-data/derivatives/pipeline-cross*")
folders.sort()
subject = '109123'
estimators = []
for f in folders:
    folder = os.path.join(f, subject)
    fname = glob.glob(folder+'/*pickle')[0]
    targets = get_dictionary(fname)['targets'].split("+")
    session = get_dictionary(fname)['ses'].split("+")
    estimator = load(fname)
    estimators.append([targets, session, estimator['estimator']])


loader = DataLoader(configuration_file=conf_file, 
                    loader='bids-meg',
                    bids_window='300',
                    #bids_ses='02',
                    task='power')
ds = loader.fetch(subject_names=['sub-109123'], prepro=[Transformer()])

session_generalization_accuracy = []
for targets, ses, estimator in estimators:
    
    if ses == ['01']:
        ses = ['02']
    else:
        ses = ['01']

    ds_ = SampleSlicer(targets=targets, ses=ses).transform(ds)

    X = ds_.samples.copy()
    y_ = ds_.targets.copy()

    y = LabelEncoder().fit_transform(y_)

    accuracy = np.mean([est.score(X, y) for est in estimator], axis=0)

    session_generalization_accuracy.append(accuracy)

order = [1, 3, 2, 0]
figures = [pl.subplots(3, 4, sharex=True) for i in range(2)]

for i, (targets, ses, estimator) in enumerate(estimators):

    class_ = "-".join(targets)
    clf = str(estimator[0].base_estimator['clf'])
    kernel = ''
    if clf.find('SVC') != -1 and clf.find("kernel") != -1:
        kernel = clf[clf.find("kernel='"):clf.find(", p")]
    clf = clf[:clf.find('(')] + " " + kernel

    print(clf, class_, ses, i%4, np.floor((i/4)%2))
    
    f = np.int(np.floor((i/4)%2))
    fig, axes = figures[f]

    r = order[i%4]

    limits = (0.4, .99)


    if ses == ['01']:
        color = 'cornflowerblue'
        j = 0
    else:
        color = 'tomato'
        j = 1

    accuracy = session_generalization_accuracy[i]
    im = axes[j, r].imshow(accuracy, 
                          origin='lower',
                          cmap=pl.cm.magma,
                          vmin=limits[0],
                          vmax=limits[1]
                          )

    axes[j, r].set_title("%s | %s | %s" % (clf, class_, ses))
    axes[j, r].set_xticks(np.arange(58)[::9])
    axes[j, r].set_xticklabels(xticklabels[::9])
    axes[j, r].set_xlabel('Training time')

    axes[j, r].set_yticks(np.arange(58)[::9])
    axes[j, r].set_yticklabels(xticklabels[::9])
    axes[j, r].set_ylabel('Testing time')
    axes[j, r].vlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')
    axes[j, r].hlines(7.5, -0.2, 58.2, colors='r', linestyles='dashed')

    axes[2, r].plot(np.diag(accuracy), c=color)
    axes[2, r].set_xlabel('Training time')
    axes[2, r].set_ylabel('Classification accuracy')
    axes[2, r].set_ylim(limits)
    axes[2, r].vlines(7.5, limits[0], limits[1], colors='r', linestyles='dashed')
    
    if f == 2:
        fig.colorbar(im, ax=axes[:], location='right')