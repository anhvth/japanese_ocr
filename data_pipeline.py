import Augmentor
p = Augmentor.Pipeline(dbs=None, path_text2label=None, text_corpus='corpus/poc')
p.zoom(1, 0.5, 0.6)
p.random_contrast(1, 0.3, 1)
p.rotate(1, 3, 3)
p.sample(10, False)