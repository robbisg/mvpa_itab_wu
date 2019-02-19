from mvpa_itab.io import DataLoader
from mvpa_itab.preprocessing.pipelines import PreprocessingPipeline
from sklearn.model_selection._split import GroupShuffleSplit
from mvpa_itab.pipeline.decoding.roi_decoding import Decoding
from sklearn.pipeline import Pipeline
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.svm.classes import SVC, SVR
from mvpa_itab.preprocessing.functions import SampleSlicer,\
    FeatureWiseNormalizer, TargetTransformer
from mvpa_itab.pipeline.searchlight import SearchLight
from sklearn.preprocessing.label import LabelEncoder


loader = DataLoader(configuration_file="/media/robbis/DATA/fmri/monks/meditation.conf", task='meg')
ds = loader.fetch()

# Preprocessing
pipeline = PreprocessingPipeline(nodes=[ 
                                        SampleSlicer({'band':['alpha'], 
                                                      'condition':['vipassana']}),
                                        FeatureWiseNormalizer(),
                                        TargetTransformer("expertise_hours")
                                        ])
ds_ = pipeline.transform(ds)


# Estimator
estimator_pp = Pipeline(steps=[('svr', SVR(C=1, kernel='linear'))])

cross_validation = GroupShuffleSplit(n_splits=10, test_size=0.25)
scores = ['r2','explained_variance']
cv_attr = 'subject'


sl = SearchLight(estimator=estimator_pp, scoring=scores, cv=cross_validation)
sl.fit(ds_, cv_attr=cv_attr)


#### Cross Validation ###
cross_validation = GroupShuffleSplit(n_splits=150, test_size=0.25)
groups = LabelEncoder().fit_transform(ds_.sa.subject)
X = ds_.samples
y = LabelEncoder().fit_transform(ds_.targets)
train_list = []
for train, test in cross_validation.split(X, y, groups=groups):
    train_list.append(ds_[train].sa.subject)
    
c = Counter(np.hstack(train_list))
