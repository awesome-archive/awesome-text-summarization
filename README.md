# awesome-text-summarization

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources dedicated to text summarization

## Table of Contents

* [Corpus](#corpus)
* [Text Summarization Software](#text-summarization-software)
* [Word Representations](#word-representations)
  * [Word Representations for Chinese](#word-representations-for-chinese)
  * [Evaluation of Word Embeddings](#evaluation-of-word-embeddings)
  * [Evaluation of Word Embeddings for Chinese](#evaluation-of-word-embeddings-for-chinese)
* [Sentence Representations](#sentence-representations)
  * [Evaluation of Sentence Embeddings](#evaluation-of-sentence-embeddings)
* [Extractive Text Summarization](#extractive-text-summarization)
* [Abstractive Text Summarization](#abstractive-text-summarization)
* [Text Summarization](#text-summarization)
* [Chinese Text Summarization](#chinese-text-summarization)
* [Evaluation Metrics](#evaluation-metrics)
* [Opinion Summarization](#opinion-summarization)

## Contents

### Corpus

1. [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) contains 51 articles. Each article is about a product’s feature, like iPod’s Battery Life, etc. and is a collection of reviews by customers who purchased that product. Each article in the dataset has 5 manually written “gold” summaries. Usually the 5 gold summaries are different but they can also be the same text repeated 5 times.
2. [Past DUC Data](https://www-nlpir.nist.gov/projects/duc/data.html) and [TAC Data](https://tac.nist.gov/data/index.html) include summarization data.
3. [English Gigaword](https://catalog.ldc.upenn.edu/LDC2003T05): English Gigaword was produced by Linguistic Data Consortium (LDC).
4. [Large Scale Chinese Short Text Summarization Dataset (LCSTS)](http://icrc.hitsz.edu.cn/Article/show/139.html): This corpus is constructed from the Chinese microblogging website SinaWeibo. It consists of over 2 million real Chinese short texts with short summaries given by the writer of each text.
5. Ziqiang Cao, Chengyao Chen, Wenjie Li, Sujian Li, Furu Wei, Ming Zhou. [TGSum: Build Tweet Guided Multi-Document Summarization Dataset](https://arxiv.org/abs/1511.08417v1). arXiv:1511.08417, 2015.
6. [scisumm-corpus](https://github.com/WING-NUS/scisumm-corpus) contains a release of the scientific document summarization corpus and annotations from the WING NUS group.
7. Avinesh P.V.S., Maxime Peyrard, Christian M. Meyer. [Live Blog Corpus for Summarization](https://arxiv.org/abs/1802.09884v1). arXiv:1802.09884, 2018.
8. Alexander R. Fabbri, Irene Li, Prawat Trairatvorakul, Yijiao He, Wei Tai Ting, Robert Tung, Caitlin Westerfield, Dragomir R. Radev.[TutorialBank: A Manually-Collected Corpus for Prerequisite Chains, Survey Extraction and Resource Recommendation](https://arxiv.org/abs/1805.04617). arXiv:1805.04617, 2018. The source code is [TutorialBank](https://github.com/Yale-LILY/TutorialBank). All the datasets could be found through the [search engine](http://tangra.cs.yale.edu/newaan/). The blog [TutorialBank: Learning NLP Made Easier](https://alex-fabbri.github.io/TutorialBank/) is an excellent user guide with step by step instructions on how to use the search engine.
9. [Legal Case Reports Data Set](https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports) contains Australian legal cases from the Federal Court of Australia (FCA).
10. [TIPSTER Text Summarization Evaluation Conference (SUMMAC)](https://www-nlpir.nist.gov/related_projects/tipster_summac/cmp_lg.html)  includes 183 documents.
11. [NEWS SUMMARY](https://www.kaggle.com/sunnysai12345/news-summary) consists of 4515 examples.
12. [BBC News Summary](https://www.kaggle.com/pariza/bbc-news-summary/data) consists of 417 political news articles of BBC from 2004 to 2005.
13. CNN / Daily Mail dataset (non-anonymized) for summarization is produced by the code [cnn-dailymail](https://github.com/abisee/cnn-dailymail).
14. [sentence-compression](https://github.com/google-research-datasets/sentence-compression) is a large corpus of uncompressed and compressed sentences from news articles. The algorithm to collect the data is described here: [Overcoming the Lack of Parallel Data in Sentence Compression](http://www.aclweb.org/anthology/D/D13/D13-1155.pdf) by Katja Filippova and Yasemin Altun, EMNLP '13.
15. [The Columbia Summarization Corpus (CSC)](https://www.cs.ucsb.edu/~william/papers/ijcnlp2011.pdf) was retrieved from the output of the [Newsblaster online news summarization system](http://newsblaster.cs.columbia.edu/) that crawls the Web for news articles, clusters them on specific topics and produces multidocument summaries for each cluster. They collected a total of 166,435 summaries containing 2.5 million sentences and covering 2,129 days in the 2003-2011 period. Additional references of the Columbia Newsblaster summarizer can be found on the website of [Columbia NLP group publication page](http://www1.cs.columbia.edu/nlp/papers.cgi).
16. [WikiHow-Dataset](https://github.com/mahnazkoupaee/WikiHow-Dataset)  a new large-scale dataset using the online [WikiHow] (http://www.wikihow.com) knowledge base. Each article consists of multiple paragraphs and each paragraph starts with a sentence summarizing it. By merging the paragraphs to form the article and the paragraph outlines to form the summary, the resulting version of the dataset contains more than 200,000 long-sequence pairs.


### Text Summarization Software

1. [sumeval](https://github.com/chakki-works/sumeval) implemented in Python is a well tested & multi-language evaluation framework for text summarization.
2. [sumy](https://github.com/miso-belica/sumy) is a simple library and command line utility for extracting summary from HTML pages or plain texts. The package also contains simple evaluation framework for text summaries. Implemented summarization methods are *Luhn*, *Edmundson*, *LSA*, *LexRank*, *TextRank*, *SumBasic* and *KL-Sum*.
3. [TextRank4ZH](https://github.com/letiantian/TextRank4ZH) implements the *TextRank* algorithm to extract key words/phrases and text summarization
in Chinese. It is written in Python.
4. [snownlp](https://github.com/isnowfy/snownlp) is python library for processing Chinese text.
5. [PKUSUMSUM](https://github.com/PKULCWM/PKUSUMSUM) is an integrated toolkit for automatic document summarization. It supports single-document, multi-document and topic-focused multi-document summarizations, and a variety of summarization methods have been implemented in the toolkit. It supports Western languages (e.g. English) and Chinese language.
6. [fnlp](https://github.com/FudanNLP/fnlp) is a toolkit for Chinese natural language processing.
7. [fairseq](https://github.com/pytorch/fairseq) is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks. It provides reference implementations of various sequence-to-sequence model.

### Word Representations

1. G. E. Hinton, J. L, McClelland, and D. E. Rumelhart. [Distributed representations](https://web.stanford.edu/~jlmcc/papers/PDP/Chapter3.pdf). In D. E. Rumelhart and J. L. McClelland, Parallel Distributed Processing: Explorations in the Microstructure of Cognition. Volume 1: Foundations, MIT Press, Cambridge, MA. 1986. The related slides are [here](http://www.cs.toronto.edu/~bonner/courses/2014s/csc321/lectures/lec5.pdf) or [here](http://www.cs.toronto.edu/~bonner/courses/2016s/csc321/lectures/extra/coarse.pdf).
   * "Distributed representation" means a many-tomany relationship between two types of representation (such as concepts and neurons): 1. Each concept is represented by many
   neurons; 2. Each neuron participates in the representation of many concepts.
2. [Language Modeling with N-Grams](https://web.stanford.edu/~jurafsky/slp3/4.pdf). The related slides are [here](https://web.stanford.edu/~jurafsky/slp3/slides/LM_4.pptx). It introduced language modeling and the N-gram, one of the most widely used tools in language processing.
   * Language models offer a way to assign a probability to a sentence or other sequence of words, and to predict a word from preceding words.
   * N-grams are Markov models that estimate words from a fixed window of previous words. N-gram probabilities can be estimated by counting in a corpus and normalizing (the maximum likelihood estimate).
   * N-gram language models are evaluated extrinsically in some task, or intrinsically using perplexity.
   * The perplexity of a test set according to a language model is the geometric mean of the inverse test set probability computed by the model.
   * Smoothing algorithms provide a more sophisticated way to estimat the probability of N-grams. Commonly used smoothing algorithms for N-grams rely on lower-order N-gram counts through backoff or interpolation.
   * There are at least two drawbacks for the n-gram language model. First, it is not taking into account contexts farther than 1 or 2 words. N-grams with n up to 5 (i.e. 4 words of context) have been reported, though, but due to data scarcity, most predictions are made with a much shorter context. Second, it is not taking into account the “similarity” between words. 
3. Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin. [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Journal of Machine Learning Research, 2003.
   - They propose continuous space LMs using neural networks to fight the curse of dimensionality by learning a distributed representation for words.
   - The model learns simultaneously (1) a distributed representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations.
   - Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence.
   - The idea of the proposed approach can be summarized: 1. associate with each word in the vocabulary a distributed word feature vector, 2. express the joint probability function of word sequences in terms of the feature vectors of these words in the sequence, and 3. learn simultaneously the word feature vectors and the parameters of that probability function.
4. In the following two papers, it is shown that both to project all words of the context onto a continuous space and calculate the language model probability for the given context can be performed by a neural network using two hidden layers.
   * Holger Schwenk and Jean-Luc Gauvain. [Training Neural Network Language Models On Very Large Corpora](ftp://tlp.limsi.fr/public/emnlp05.pdf). in Proc. Joint Conference HLT/EMNLP, 2005.
   * Holger Schwenk. [Continuous space language models](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2_2009_10/sdarticle.pdf). Computer Speech and Language, 2007.
5. Tomas Mikolov's series of papers improved the quality of word representations:
   * T. Mikolov, J. Kopecky, L. Burget, O. Glembek and J. Cernocky. [Neural network based language models for higly inflective languages](http://www.fit.vutbr.cz/research/groups/speech/publi/2009/mikolov_ic2009_nnlm_4.pdf). Proc. ICASSP, 2009. The first step in their architecture is training of bigram neural network: given word w from vocabulary V, estimate probability distribution of the next word in text. To compute projection of word w onto a  continuous space, half of the bigram network (first two layers) is used to compute values in hidden layer. Values from the hidden layer of bigram network are used to form input layer of n-gram network.
   * T. Mikolov, W.T. Yih and G. Zweig. [Linguistic Regularities in Continuous Space Word Representations](https://www.aclweb.org/anthology/N13-1090). NAACL HLT, 2013. They examine the vector-space word representations that are implicitly learned by the input-layer weights. They find that these representations are surprisingly good at capturing syntactic and semantic regularities in language, and that each relationship is characterized by a relation-specific vector offset. This allows vector-oriented reasoning based on the offsets between words. Remarkably, this method outperforms the best previous systems.
   * Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781v3). arXiv:1301.3781v3, 2013. They propose two new model architectures for learning distributed representations: 1. Continuous Bag-of-Words Model (CBOW) builds a log-linear classifier with context words at the input, where the training criterion is to correctly classify the current word; 2. Continuous Skip-gram Model uses each current word as an input to a log-linear classifier with continuous projection layer, and predicts words within a certain range before and after the current word.
   * Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado and Jeffrey Dean. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546). arXiv:1310.4546, 2013. The source code written in C is [word2vec](https://github.com/tmikolov/word2vec). They present several extensions of the original Skip-gram model. They show that sub-sampling of frequent words during training results in a significant speedup (around 2x - 10x), and improves accuracy of the representations of less frequent words. In addition, they present a simplified variant of Noise Contrastive Estimation for training the Skip-gram model that results in faster training and better vector representations for frequent words. Word based model is extended to phrase based model. They found that simple vector addition can often produce meaningful results.
   * Tomas Mikolov, Edouard Grave, Piotr Bojanowski, Christian Puhrsch and Armand Joulin.[Advances in Pre-Training Distributed Word Representations](https://arxiv.org/abs/1712.09405). arXiv:1712.09405, 2017. They show that several modifications of the standard word2vec training pipeline significantly improves the quality of the resulting word vectors: position-dependent weighting, the phrase representations and the subword information.
6. Christopher Olah. [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/). This post reviews some extremely remarkable results in applying deep neural networks to NLP, where the representation perspective of deep learning is a powerful view that seems to answer why deep neural networks are so effective. 
7. Levy, Omer, and Yoav Goldberg. [Neural word embedding as implicit matrix factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf). NIPS. 2014.
8. [Sanjeev Arora](http://www.offconvex.org/)'s a series of blogs/papers about word embeddings:
   * The blog [Semantic Word Embeddings](http://www.offconvex.org/2015/12/12/word-embeddings-1/) is a very good overview about word embedding.
   * The blog [Word Embeddings: Explaining their properties](http://www.offconvex.org/2016/02/14/word-embeddings-2/) introduces the main result about [RAND-WALK: A Latent Variable Model Approach to Word Embeddings](https://arxiv.org/abs/1502.03520), which answers three interesting questions: 1. Why do low-dimensional embeddings capture huge statistical information? 2. Why do low dimensional embeddings work better than high-dimensional ones? 3. Why do Semantic Relations correspond to Directions?
   * The blog [Linear algebraic structure of word meanings](http://www.offconvex.org/2016/07/10/embeddingspolysemy/) introduces the main result about [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://arxiv.org/abs/1601.03764), which shows that word senses are easily accessible in many current word embeddings.
9. [Word2Vec Resources](http://mccormickml.com/2016/04/27/word2vec-resources/): This is a post with links to and descriptions of word2vec tutorials, papers, and implementations.
10. [Word embeddings: how to transform text into numbers](https://monkeylearn.com/blog/word-embeddings-transform-text-numbers/)
11. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus.
12. Li, Yitan, et al. [Word embedding revisited: A new representation learning and explicit matrix factorization perspective](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.719.9973&rep=rep1&type=pdf). IJCAI. 2015.
13. O. Levy, Y. Goldberg, and I. Dagan. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016). Trans. Assoc. Comput. Linguist., 2015.
14. Eric Nalisnick, Sachin Ravi. [Learning the Dimensionality of Word Embeddings](https://arxiv.org/abs/1511.05392v3). arXiv:1511.05392, 2015.
    * They describe a method for learning word embeddings with data-dependent dimensionality. Their Stochastic Dimensionality Skip-Gram (SD-SG) and Stochastic Dimensionality Continuous Bag-of-Words (SD-CBOW) are nonparametric analogs of Mikolov et al.'s (2013) well-known 'word2vec' model.
15. William L. Hamilton, Jure Leskovec, Dan Jurafsky. [Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change](https://aclanthology.info/pdf/P/P16/P16-1141.pdf).
    * Hamilton et al. model changes in word meaning by fitting word embeddings on consecutive corpora of historical language. They compare several ways of quantifying meaning (co-occurrence vectors weighted by PPMI, SVD embeddings and word2vec embeddings), and align historical embeddings from different corpora by finding the optimal rotational alignment that preserves the cosine similarities as much as possible.
16. Zijun Yao, Yifan Sun, Weicong Ding, Nikhil Rao, Hui Xiong. [Dynamic Word Embeddings for Evolving Semantic Discovery](https://arxiv.org/abs/1703.00607). arXiv:1703.00607v2, International Conference on Web Search and Data Mining (WSDM 2018).
17. Yang, Wei  and  Lu, Wei  and  Zheng, Vincent. [A Simple Regularization-based Algorithm for Learning Cross-Domain Word Embeddings](http://www.aclweb.org/anthology/D/D17/D17-1312.pdf). ACL, 2017. The source code in C is [cross_domain_embedding](https://github.com/Victor0118/cross_domain_embedding).
    - This paper presents a simple yet effective method for learning word embeddings based on text from different domains.
18. Sebastian Ruder. [Word embeddings in 2017: Trends and future directions](http://ruder.io/word-embeddings-2017/)
19. Bryan McCann, James Bradbury, Caiming Xiong and Richard Socher. [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107). For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors). 
    * A Keras/TensorFlow implementation of the MT-LSTM/CoVe is [CoVe](https://github.com/rgsachin/CoVe).
    * A PyTorch implementation of the MT-LSTM/CoVe is [cove](https://github.com/salesforce/cove).
20. Maria Pelevina, Nikolay Arefyev, Chris Biemann, Alexander Panchenko. [Making Sense of Word Embeddings](https://arxiv.org/abs/1708.03390). arXiv:1708.03390, 2017. The source code written in Python is [sensegram](https://github.com/tudarmstadt-lt/sensegram).
    - Making sense embedding out of word embeddings using graph-based word sense induction.
21. Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606v2). arXiv:1607.04606v2, 2017. The souce code in C++11 is [fastText](https://fasttext.cc/), which is a library for efficient learning of word representations and sentence classification.
    * They propose a new approach based on the skipgram model, where each word is represented as a bag of character n-grams. A vector representation is associated to each character n-gram; words being represented as the sum of these representations. 
22. Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer and Herv{\'e} J{\'e}gou. [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087). arXiv:1710.04087, 2017. The source code in Python is [MUSE](https://github.com/facebookresearch/MUSE), which is a library for multilingual unsupervised or supervised word embeddings.
23. Gabriel Grand, Idan Asher Blank, Francisco Pereira, Evelina Fedorenko. [Semantic projection: recovering human knowledge of multiple, distinct object features from word embeddings](https://arxiv.org/abs/1802.01241v2). arXiv:1802.01241, 2018.
    * Could context-dependent relationships be recovered from word embeddings? To address this issue, they introduce a powerful, domain-general solution: "semantic projection" of word-vectors onto lines that represent various object features, like size (the line extending from the word "small" to "big"), intelligence (from "dumb" to "smart"), or danger (from "safe" to "dangerous").
24. Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365). arXiv:1802.05365, NAACL 2018. The code is [ELMo](https://allennlp.org/elmo).
25. Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, Tomas Mikolov. [Learning Word Vectors for 157 Languages](https://arxiv.org/abs/1802.06893v2). arXiv:1802.06893v2, Proceedings of LREC, 2018.
    * They describe how high quality word representations for 157 languages are trained. They used two sources of data to train these models: the free online encyclopedia Wikipedia and data from the common crawl project. Pre-trained word vectors for 157 languages are [available](https://fasttext.cc/docs/en/crawl-vectors.html).
26. Douwe Kiela, Changhan Wang and Kyunghyun Cho. [Context-Attentive Embeddings for Improved Sentence Representations](https://arxiv.org/abs/1804.07983). arXiv:1804.07983, 2018. 
    * While one of the first steps in many NLP systems is selecting what embeddings to use, they argue that such a step is better left for neural networks to figure out by themselves. To that end, they introduce a novel, straightforward yet highly effective method for combining multiple types of word embeddings in a single model, leading to state-of-the-art performance within the same model class on a variety of tasks.
27. Laura Wendlandt, Jonathan K. Kummerfeld, Rada Mihalcea. [Factors Influencing the Surprising Instability of Word Embeddings](https://arxiv.org/abs/1804.09692v1). arXiv:1804.09692, NAACL HLT 2018.
    * They provide empirical evidence for how various factors contribute to the stability of word embeddings, and analyze the effects of stability on downstream tasks.
28. [magnitude](https://github.com/plasticityai/magnitude) is a feature-packed Python package and vector storage file format for utilizing vector embeddings in machine learning models in a fast, efficient, and simple manner.

#### Word Representations for Chinese

1. X. Chen, L.  Xu, Z.  Liu, M. Sun and H. Luan. [Joint Learning of Character and Word Embeddings](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf). IJCAI, 2015. The source code in C is [CWE](https://github.com/Leonard-Xu/CWE).
2. Jian Xu, Jiawei Liu, Liangang Zhang, Zhengyu Li, Huanhuan Chen. [Improve Chinese Word Embeddings by Exploiting Internal Structure](http://www.aclweb.org/anthology/N16-1119). NAACL 2016. The source code in C is [SCWE](https://github.com/JianXu123/SCWE).
3. Jinxing Yu, Xun Jian, Hao Xin and Yangqiu Song. [Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components](http://www.cse.ust.hk/~yqsong/papers/2017-EMNLP-ChineseEmbedding.pdf). EMNLP, 2017. The source code in C is [JWE](https://github.com/HKUST-KnowComp/JWE).
4. Shaosheng Cao and Wei Lu. [Improving Word Embeddings with Convolutional Feature Learning and Subword Information](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPDFInterstitial/14724/14187). AAAI, 2017. The source code in C# is [IWE](https://github.com/ShelsonCao/IWE).
5. Zhe Zhao, Tao Liu, Shen Li, Bofang Li and Xiaoyong Du. [Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://aclweb.org/anthology/D17-1023).  EMNLP, 2017. The source code in Python is [ngram2vec](https://github.com/zhezhaoa/ngram2vec).
6. Shaosheng Cao, Wei Lu, Jun Zhou, Xiaolong Li. [cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17444). AAAI, 2018. The source code in C++ is [cw2vec](https://github.com/bamtercelboo/cw2vec). 

#### Evaluation of Word Embeddings

1. Tobias Schnabel, Igor Labutov, David Mimno and Thorsten Joachims. [Evaluation methods for unsupervised word embeddings](https://www.cs.cornell.edu/~schnabts/downloads/schnabel2015embeddings.pdf). EMNLP, 2015. The slides are [here](https://www.cs.cornell.edu/~schnabts/downloads/slides/schnabel2015eval.pdf).
2. Stanisław Jastrzebski, Damian Leśniak, Wojciech Marian Czarnecki. [How to evaluate word embeddings? On importance of data efficiency and simple supervised tasks](https://arxiv.org/abs/1702.02170). arXiv:1702.02170, 2017. The source code in Python is [word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks).
3. Amir Bakarov. [A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/abs/1801.09536). arXiv:1801.09536, 2018.

#### Evaluation of Word Embeddings for Chinese

1. Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du. [Analogical Reasoning on Chinese Morphological and Semantic Relations](https://arxiv.org/abs/1805.06504). arXiv:1805.06504, ACL, 2018. 
   * The project [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors) provides 100+ Chinese Word Embeddings trained with different representations (dense and sparse), context features (word, ngram, character, and more), and corpora. Moreover, it provides a Chinese analogical reasoning dataset CA8 and an evaluation toolkit for users to evaluate the quality of their word vectors.

### Sentence Representations

1. Kalchbrenner, Nal, Edward Grefenstette, and Phil Blunsom. [A convolutional neural network for modelling sentences](http://arxiv.org/abs/1404.2188). arXiv:1404.2188, 2014.
2. Quoc Le and Tomas Mikolov. [Distributed representations of sentences and documents](https://arxiv.org/abs/1405.4053v2). arXiv:1405.4053v2, 2014.
   * Distributed Memory Model of Paragraph Vectors (PV-DM): The inspiration is that the paragraph vectors are asked to contribute to the prediction task of the next word given many contexts sampled from the paragraph.  The paragraph vector and word vectors are averaged or concatenated to predict the next word in a context. The contexts are fixed-length and sampled from a sliding window over the paragraph. The paragraph vector is shared across all contexts generated from the same paragraph but not across paragraphs. However, the word vector matrix is shared across paragraphs. The downside is at prediction time, inference needs to be performed to compute a new vector.
   * Distributed Bag of Words version of Paragraph Vector (PV-DBOW): This modle is to ignore the context words in the input, but force the model to predict words randomly sampled from the paragraph in the output.
3. Yoon Kim. [Convolutional neural networks for sentence classification](http://arxiv.org/abs/1408.5882). arXiv:1408.5882, EMNLP 2014.
1. Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun and Sanja Fidler. [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726). arXiv:1506.06726, 2015. The source code in Python is [skip-thoughts](https://github.com/ryankiros/skip-thoughts). The TensorFlow implementation of *Skip-Thought Vectors* is [skip_thoughts](https://github.com/tensorflow/models/tree/master/research/skip_thoughts)
   * Instead of using a word to predict its surrounding context, they instead encode a sentence to predict the sentences around it. The skip-thoughts is in the framework of encoder-decoder models: an encoder maps words to a sentence vector and a decoder is used to generate the surrounding sentences.
   * The end product of skip-thoughts is the encoder,  which can then be used to generate fixed length representations of sentences. The decoders are thrown away after training.
   * A good tutorial to this paper is [My Thoughts On Skip Thoughts](http://sanyam5.github.io/my-thoughts-on-skip-thoughts/).
2. John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu. [Towards Universal Paraphrastic Sentence Embeddings](https://arxiv.org/abs/1511.08198). arXiv:1511.08198, ICLR 2016. The source code written in Python is [iclr2016](https://github.com/jwieting/iclr2016).
2. Zhe Gan, Yunchen Pu, Ricardo Henao, Chunyuan Li, Xiaodong He, Lawrence Carin. [Learning Generic Sentence Representations Using Convolutional Neural Networks](https://arxiv.org/abs/1611.07897). arXiv:1611.07897, EMNLP 2017. The training code written in Python is [ConvSent](https://github.com/zhegan27/ConvSent).
3. Matteo Pagliardini, Prakhar Gupta, Martin Jaggi. [Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/abs/1703.02507v2). arXiv:1703.02507, NAACL 2018. The source code in Python is [sent2vec](https://github.com/epfml/sent2vec). 
1. Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, Jason Weston. [StarSpace: Embed All The Things](https://arxiv.org/abs/1709.03856v5). arXiv:1709.03856v5, 2017. The source code in C++11 is [StarSpace](https://github.com/facebookresearch/Starspace/).
2. Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, Antoine Bordes. [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364v5). arXiv:1705.02364v5, EMNLP 2017. The source code in Python is [InferSent](https://github.com/facebookresearch/InferSent).
3. Sanjeev Arora, Yingyu Liang, Tengyu Ma. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx). ICLR 2017. The source code written in Python is [SIF](https://github.com/PrincetonML/SIF). [SIF_mini_demo](https://github.com/PrincetonML/SIF_mini_demo) is a minimum example for the sentence embedding algorithm. [sentence2vec](https://github.com/peter3125/sentence2vec) is another implementation.
1. Yixin Nie, Mohit Bansal. [Shortcut-Stacked Sentence Encoders for Multi-Domain Inference](https://arxiv.org/abs/1708.02312). arXiv:1708.02312, EMNLP 2017. The source code in Python is [multiNLI_encoder](https://github.com/easonnie/multiNLI_encoder). The new repo [ResEncoder]( https://github.com/easonnie/ResEncoder) is for Residual-connected sentence encoder for NLI.
2. Allen Nie, Erin D. Bennett, Noah D. Goodman. [DisSent: Sentence Representation Learning from Explicit Discourse Relations](https://arxiv.org/abs/1710.04334v2). arXiv:1710.04334v2, 2018.
3. Andreas Rücklé, Steffen Eger, Maxime Peyrard, Iryna Gurevych. [Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations](https://arxiv.org/abs/1803.01400v2).  arXiv:1803.01400v2, 2018. The source code written in Python is [arxiv2018-xling-sentence-embeddings](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings).
2. Lajanugen Logeswaran, Honglak Lee. [An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893). arXiv:1803.02893, ICLR 2018. The open review comments are listed [here](https://openreview.net/forum?id=rJvJXZb0W).
3. Eric Zelikman. [Context is Everything: Finding Meaning Statistically in Semantic Spaces](https://arxiv.org/abs/1803.08493). arXiv:1803.08493, 2018.
1. Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, Ray Kurzweil. [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175v2). arXiv:1803.11175v2, 2018.
2. Sandeep Subramanian, Adam Trischler, Yoshua Bengio, Christopher J Pal. [Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning](https://arxiv.org/abs/1804.00079). arXiv:1804.00079, ICLR 2018.
3. [LASER](https://github.com/facebookresearch/LASER) is a library to calculate multilingual sentence embeddings:
   * Holger Schwenk and Matthijs Douze. [Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://aclanthology.info/papers/W17-2619/w17-2619). ACL workshop on Representation Learning for NLP, 2017.
   * Holger Schwenk and Xian Li. [A Corpus for Multilingual Document Classification in Eight Languages](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf).  LREC, 2018.
   * Holger Schwenk. [Filtering and Mining Parallel Data in a Joint Multilingual Space](https://arxiv.org/abs/1805.09822). arXiv:1805.09822, ACL, 2018.

#### Evaluation of Sentence Embeddings

1. Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer Lavi, Yoav Goldberg. [Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks](https://arxiv.org/abs/1608.04207v3). arXiv:1608.04207v3, 2017.
   * They define prediction tasks around isolated aspects of sentence structure (namely sentence length, word content, and word order), and score representations by the ability to train a classifier to solve each prediction task when using the representation as input.
2. Alexis Conneau, Douwe Kiela. [SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://arxiv.org/abs/1803.05449). arXiv:1803.05449, LREC 2018. The source code in Python is [SentEval](https://github.com/facebookresearch/SentEval). **SentEval** encompasses a variety of tasks, including binary and multi-class classification, natural language inference and sentence similarity.
3. Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman. [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461). arXiv:1804.07461, 2018.
4. Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, Marco Baroni. [What you can cram into a single vector: Probing sentence embeddings for linguistic properties](https://arxiv.org/abs/1805.01070v2). arXiv:1805.01070v2, 2018.

### Extractive Text Summarization

1. H. P. Luhn. [The automatic creation of literature abstracts](http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf). IBM Journal of Research and Development, 1958. Luhn's method is as follows:
   1. Ignore Stopwords: Common words (known as stopwords) are ignored.
   2. Determine Top Words: The most often occuring words in the document are counted up.
   3. Select Top Words: A small number of the top words are selected to be used for scoring.
   4. Select Top Sentences: Sentences are scored according to how many of the top words they contain. The top four sentences are selected for the summary.
2. H. P. Edmundson. [New Methods in Automatic Extracting](http://courses.ischool.berkeley.edu/i256/f06/papers/edmonson69.pdf). Journal of the Association for Computing Machinery, 1969.
3. David M. Blei, Andrew Y. Ng and Michael I. Jordan. [Latent Dirichlet Allocation](http://ai.stanford.edu/~ang/papers/jair03-lda.pdf). Journal of Machine Learning Research, 2003. The source code in Python is [sklearn.decomposition.LatentDirichletAllocation](http://scikit-learn.org/dev/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html). Reimplement Luhn's algorithm, but with topics instead of words and applied to several documents instead of one.
   1. Train LDA on all products of a certain type (e.g. all the books)
   2. Treat all the reviews of a particular product as one document, and infer their topic distribution
   3. Infer the topic distribution for each sentence
   4. For each topic that dominates the reviews of a product, pick some sentences that are themselves dominated by that topic.
4. David M. Blei. [Probabilistic Topic Models](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf). Communications of the ACM, 2012.
5. Rada Mihalcea and Paul Tarau. [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf). ACL, 2004. The source code in Python is [pytextrank](https://github.com/ceteri/pytextrank). `pytextrank` works in four stages, each feeding its output to the next:
   - Part-of-Speech Tagging and lemmatization are performed for every sentence in the document.
   - Key phrases are extracted along with their counts, and are normalized.
   - Calculates a score for each sentence by approximating jaccard distance between the sentence and key phrases.
   - Summarizes the document based on most significant sentences and key phrases.
6. Federico Barrios, Federico López, Luis Argerich and Rosa Wachenchauzer. [Variations of the Similarity Function of TextRank for Automated Summarization](https://arxiv.org/abs/1602.03606). arXiv:1602.03606, 2016. The source code in Python is [gensim.summarization](http://radimrehurek.com/gensim/). Gensim's summarization only works for English for now, because the text is pre-processed so that stop words are removed and the words are stemmed, and these processes are language-dependent. TextRank works as follows:
   - Pre-process the text: remove stop words and stem the remaining words.
   - Create a graph where vertices are sentences.
   - Connect every sentence to every other sentence by an edge. The weight of the edge is how similar the two sentences are.
   - Run the PageRank algorithm on the graph.
   - Pick the vertices(sentences) with the highest PageRank score.
7. [TextTeaser](https://github.com/MojoJolo/textteaser) uses basic summarization features and build from it. Those features are:
   - Title feature is used to score the sentence with the regards to the title. It is calculated as the count of words which are common to title of the document and sentence.
   - Sentence length is scored depends on how many words are in the sentence. TextTeaser defined a constant “ideal” (with value 20), which represents the ideal length of the summary, in terms of number of words. Sentence length is calculated as a normalized distance from this value.
   - Sentence position is where the sentence is located. I learned that introduction and conclusion will have higher score for this feature.
   - Keyword frequency is just the frequency of the words used in the whole text in the bag-of-words model (after removing stop words).
8. Güneş Erkan and Dragomir R. Radev. [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html). 2004.
   - LexRank uses IDF-modified Cosine as the similarity measure between two sentences. This similarity is used as weight of the graph edge between two sentences. LexRank also incorporates an intelligent post-processing step which makes sure that top sentences chosen for the summary are not too similar to each other.
9. [Latent Semantic Analysis(LSA) Tutorial](https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/).
10. Josef Steinberger and Karel Jezek. [Using Latent Semantic Analysis in Text Summarization and Summary Evaluation](http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf). Proc. ISIM’04, 2004.
11. Josef Steinberger and Karel Ježek. [Text summarization and singular value decomposition](https://www.researchgate.net/profile/Karel_Jezek2/publication/226424326_Text_Summarization_and_Singular_Value_Decomposition/links/57233c1308ae586b21d87e66/Text-Summarization-and-Singular-Value-Decomposition.pdf). International Conference on Advances in Information Systems, 2004.
12. Josef Steinberger, Massimo Poesio, Mijail A Kabadjov and Karel Ježek. [Two uses of anaphora resolution in summarization](http://www.sensei-conversation.eu/wp-content/uploads/files/IPMpaper_official.pdf). Information Processing & Management, 2007.
13. James Clarke and Mirella Lapata. [Modelling Compression with Discourse Constraints](http://jamesclarke.net/media/papers/clarke-lapata-emnlp07.pdf). EMNLP-CoNLL, 2007.
14. Dan Gillick and Benoit Favre. [A Scalable Global Model for Summarization](https://pdfs.semanticscholar.org/a1a2/748e68d019815f1107fa19b0ab628b63928a.pdf). ACL, 2009.
15. Ani Nenkova and Kathleen McKeown. [Automatic summarization](https://www.cis.upenn.edu/~nenkova/1500000015-Nenkova.pdf).
Foundations and Trend in Information Retrieval, 2011. [The slides](https://www.fosteropenscience.eu/sites/default/files/pdf/2932.pdf) are also available.
16. Vahed Qazvinian, Dragomir R. Radev, Saif M. Mohammad, Bonnie Dorr, David Zajic, Michael Whidby, Taesun Moon. [Generating Extractive Summaries of Scientific Paradigms](https://arxiv.org/abs/1402.0556v1). arXiv:1402.0556, 2014.
17. Kågebäck, Mikael, et al. [Extractive summarization using continuous vector space models](http://www.aclweb.org/anthology/W14-1504). Proceedings of the 2nd Workshop on Continuous Vector Space Models and their Compositionality (CVSC)@ EACL. 2014.
18. Katja Filippova, Enrique Alfonseca, Carlos A. Colmenares, Lukasz Kaiser, Oriol Vinyals. [Sentence Compression by Deletion with LSTMs](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43852.pdf). EMNLP 2015.
18. Ramesh Nallapati, Bowen Zhou, Mingbo Ma. [Classify or Select: Neural Architectures for Extractive Document Summarization](https://arxiv.org/abs/1611.04244).  arXiv:1611.04244. 2016.
19. Liangguo Wang, Jing Jiang, Hai Leong Chieu, Chen Hui Ong, Dandan Song, Lejian Liao. [Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains](http://www.aclweb.org/anthology/P17-1127). ACL 2017.
19. Ramesh Nallapati, Feifei Zhai, Bowen Zhou. [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230). arXiv:1611.04230, AAAI 2017.
20. Shashi Narayan, Nikos Papasarantopoulos, Mirella Lapata, Shay B. Cohen. [Neural Extractive Summarization with Side Information](https://arxiv.org/abs/1704.04530). arXiv:1704.04530, 2017.
21. Rakesh Verma, Daniel Lee. [Extractive Summarization: Limits, Compression, Generalized Model and Heuristics](https://arxiv.org/abs/1704.05550v1). arXiv:1704.05550, 2017.
22. Ed Collins, Isabelle Augenstein, Sebastian Riedel. [A Supervised Approach to Extractive Summarisation of Scientific Papers](https://arxiv.org/abs/1706.03946v1). arXiv:1706.03946, 2017.
23. Sukriti Verma, Vagisha Nidhi. [Extractive Summarization using Deep Learning](https://arxiv.org/abs/1708.04439v1). arXiv:1708.04439, 2017.
24. Parth Mehta, Gaurav Arora, Prasenjit Majumder. [Attention based Sentence Extraction from Scientific Articles using Pseudo-Labeled data](https://arxiv.org/abs/1802.04675v1).     arXiv:1802.04675, 2018.
25. Shashi Narayan, Shay B. Cohen, Mirella Lapata. [Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/abs/1802.08636v1). arXiv:1802.08636, NAACL, 2018.
26. Aakash Sinha, Abhishek Yadav, Akshay Gahlot. [Extractive Text Summarization using Neural Networks](https://arxiv.org/abs/1802.10137v1). arXiv:1802.10137, 2018.
27. Yuxiang Wu, Baotian Hu. [Learning to Extract Coherent Summary via Deep Reinforcement Learning](https://arxiv.org/abs/1804.07036). arXiv:1804.07036, AAAI, 2018.
28. Tanner A. Bohn, Charles X. Ling. [Neural Sentence Location Prediction for Summarization](https://arxiv.org/abs/1804.08053v1). arXiv:1804.08053, 2018.
28. Kamal Al-Sabahi, Zhang Zuping, Mohammed Nadher. [A Hierarchical Structured Self-Attentive Model for Extractive Document Summarization (HSSAS)](https://arxiv.org/abs/1805.07799v1). arXiv:1805.07799, IEEE Access, 2018.
28. Sansiri Tarnpradab, Fei Liu, Kien A. Hua. [Toward Extractive Summarization of Online Forum Discussions via Hierarchical Attention Networks](https://arxiv.org/abs/1805.10390v1). 2018.
28. Kristjan Arumae, Fei Liu. [Reinforced Extractive Summarization with Question-Focused Rewards](https://arxiv.org/abs/1805.10392v2). arXiv:1805.10392, 2018.
28. Qingyu Zhou, Nan Yang, Furu Wei, Shaohan Huang, Ming Zhou, Tiejun Zhao. [Neural Document Summarization by Jointly Learning to Score and Select Sentences](https://arxiv.org/abs/1807.02305v1). arXiv:1807.02305, ACL 2018.
28. Xingxing Zhang, Mirella Lapata, Furu Wei, Ming Zhou. [Neural Latent Extractive Document Summarization](https://arxiv.org/abs/1808.07187v2). arXiv:1808.07187, EMNLP 2018.

### Abstractive Text Summarization

1. Alexander M. Rush, Sumit Chopra, Jason Weston. [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685). EMNLP, 2015. The source code in LUA Torch7 is [NAMAS](https://github.com/facebook/NAMAS).
   - They use sequence-to-sequence encoder-decoder LSTM with attention.
   - They use the first sentence of a document. The source document is quite small (about 1 paragraph or ~500 words in the training dataset of Gigaword) and the produced output is also very short (about 75 characters). It remains an open challenge to scale up these limits - to produce longer summaries over multi-paragraph text input (even good LSTM models with attention models fall victim to vanishing gradients when the input sequences become longer than a few hundred items).
   - The evaluation method used for automatic summarization has traditionally been the ROUGE metric - which has been shown to correlate well with human judgment of summary quality, but also has a known tendency to encourage "extractive" summarization - so that using ROUGE as a target metric to optimize will lead a summarizer towards a copy-paste behavior of the input instead of the hoped-for reformulation type of summaries.
2. Peter Liu and Xin Pan. [Sequence-to-Sequence with Attention Model for Text Summarization](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html). 2016. The source code in Python is [textsum](https://github.com/tensorflow/models/tree/master/textsum).
   - They use sequence-to-sequence encoder-decoder LSTM with attention and bidirectional neural net.
   - They use the first 2 sentences of a document with a limit at 120 words.
   - The scores achieved by Google’s *textsum* are 42.57 ROUGE-1 and 23.13 ROUGE-2.
3. Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos santos, Caglar Gulcehre, Bing Xiang. [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023). arXiv:1602.06023, 2016. The souce code written in Python is [Summarization](https://github.com/zwc12/Summarization) or [abstractive-text-summarization](https://github.com/alesee/abstractive-text-summarization).
   - They use GRU with attention and bidirectional neural net.
   - They use the first 2 sentences of a documnet with a limit at 120 words.
   - They use the [Large vocabulary trick (LVT)](https://arxiv.org/abs/1412.2007) of Jean et al. 2014, which means when you decode, use only the words that appear in the source - this reduces perplexity. But then you lose the capability to do "abstractive" summary. So they do "vocabulary expansion" by adding a layer of "word2vec nearest neighbors" to the words in the input.
   - Feature rich encoding - they add TFIDF and Named Entity types to the word embeddings (concatenated) to the encodings of the words - this adds to the encoding dimensions that reflect "importance" of the words. 
   - The most interesting of all is what they call the "Switching Generator/Pointer" layer. In the decoder, they add a layer that decides to either generate a new word based on the context / previously generated word (usual decoder) or copy a word from the input (that is - add a pointer to the input). They learn when to do Generate vs. Pointer and when it is a Pointer which word of the input to Point to.
4. Konstantin Lopyrev. [Generating News Headlines with Recurrent Neural Networks](https://arxiv.org/abs/1512.01712). arXiv:1512.01712, 2015. The source code in Python is [headlines](https://github.com/udibr/headlines).
5. Jiwei Li, Minh-Thang Luong and Dan Jurafsky. [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057). arXiv:1506.01057, 2015. The source code in Matlab is [Hierarchical-Neural-Autoencoder](https://github.com/jiweil/Hierarchical-Neural-Autoencoder).
6. Sumit Chopra, Alexander M. Rush and Michael Auli. [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://harvardnlp.github.io/papers/naacl16_summary.pdf). NAACL, 2016.
7. Jianpeng Cheng, Mirella Lapata. [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/abs/1603.07252). arXiv:1603.07252, 2016.
   - This paper uses attention as a mechanism for identifying the best sentences to extract, and then go beyond that to generate an abstractive summary.
8. Siddhartha Banerjee, Prasenjit Mitra, Kazunari Sugiyama. [Generating Abstractive Summaries from Meeting Transcripts](https://arxiv.org/abs/1609.07033v1). arXiv:1609.07033, Proceedings of the 2015 ACM Symposium on Document Engineering, DocEng' 2015.
9. Siddhartha Banerjee, Prasenjit Mitra, Kazunari Sugiyama. [Multi-document abstractive summarization using ILP based multi-sentence compression](https://arxiv.org/abs/1609.07034v1). arXiv:1609.07034, 2016.
10. Suzuki, Jun, and Masaaki Nagata. [Cutting-off Redundant Repeating Generations for Neural Abstractive Summarization](http://www.aclweb.org/anthology/E17-2047).  EACL 2017 (2017): 291.
11. Jiwei Tan and Xiaojun Wan. [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](). ACL, 2017.
12. Preksha Nema, Mitesh M. Khapra, Balaraman Ravindran and Anirban Laha. [Diversity driven attention model for query-based abstractive summarization](). ACL,2017
13. Romain Paulus, Caiming Xiong, Richard Socher. [A Deep Reinforced Model for Abstractive Summarization](https://arxiv.org/abs/1705.04304). arXiv:1705.04304, 2017. The related blog is [Your tldr by an ai: a deep reinforced model for abstractive summarization](https://einstein.ai/research/your-tldr-by-an-ai-a-deep-reinforced-model-for-abstractive-summarization). 
    * Their model is trained with teacher forcing and reinforcement learning at the same time, being able to make use of both word-level and whole-summary-level supervision to make it more coherent and readable.
14. Shibhansh Dohare, Harish Karnick. [Text Summarization using Abstract Meaning Representation](https://arxiv.org/abs/1706.01678). arXiv:1706.01678, 2017.
15. Piji Li, Wai Lam, Lidong Bing, Zihao Wang. [Deep Recurrent Generative Decoder for Abstractive Text Summarization](https://arxiv.org/abs/1708.00625v1). arXiv:1708.00625, 2017.
16. Xinyu Hua, Lu Wang. [A Pilot Study of Domain Adaptation Effect for Neural Abstractive Summarization](https://arxiv.org/abs/1707.07062v1). arXiv:1707.07062, 2017.
17. Angela Fan, David Grangier, Michael Auli. [Controllable Abstractive Summarization](https://arxiv.org/abs/1711.05217v1). arXiv:1711.05217, 2017.
18. Linqing Liu, Yao Lu, Min Yang, Qiang Qu, Jia Zhu, Hongyan Li. [Generative Adversarial Network for Abstractive Text Summarization](https://arxiv.org/abs/1711.09357v1). arXiv:1711.09357, 2017.
19. Johan Hasselqvist, Niklas Helmertz, Mikael Kågebäck. [Query-Based Abstractive Summarization Using Neural Networks](https://arxiv.org/abs/1712.06100v1). arXiv:1712.06100, 2017.
20. Tal Baumel, Matan Eyal, Michael Elhadad. [Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704v2). arXiv:1801.07704, 2018.
21. André Cibils, Claudiu Musat, Andreea Hossman, Michael Baeriswyl. [Diverse Beam Search for Increased Novelty in Abstractive Summarization](https://arxiv.org/abs/1802.01457v1). arXiv:1802.01457, 2018.
22. Chieh-Teng Chang, Chi-Chia Huang, Jane Yung-Jen Hsu. [A Hybrid Word-Character Model for Abstractive Summarization](https://arxiv.org/abs/1802.09968v1). arXiv:1802.09968, 2018.
23. Asli Celikyilmaz, Antoine Bosselut, Xiaodong He, Yejin Choi. [Deep Communicating Agents for Abstractive Summarization](https://arxiv.org/abs/1803.10357v1). arXiv:1803.10357, 2018.
24. Piji Li, Lidong Bing, Wai Lam. [Actor-Critic based Training Framework for Abstractive Summarization](https://arxiv.org/abs/1803.11070v1). arXiv:1803.11070, 2018.
25. Paul Azunre, Craig Corcoran, David Sullivan, Garrett Honke, Rebecca Ruppel, Sandeep Verma, Jonathon Morgan. [Abstractive Tabular Dataset Summarization via Knowledge Base Semantic Embeddings](https://arxiv.org/abs/1804.01503v2). arXiv:1804.01503, 2018.
26. Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, Nazli Goharian. [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685v1). arXiv:1804.05685, 2018.
27. Ramakanth Pasunuru, Mohit Bansal. [Multi-Reward Reinforced Summarization with Saliency and Entailment](https://arxiv.org/abs/1804.06451v1). arXiv:1804.06451, 2018.
28. Jianmin Zhang, Jiwei Tan, Xiaojun Wan. [Towards a Neural Network Approach to Abstractive Multi-Document Summarization](https://arxiv.org/abs/1804.09010v1). arXiv:1804.09010, 2018.
28. Shuming Ma, Xu Sun, Junyang Lin, Xuancheng Ren. [A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification](https://arxiv.org/abs/1805.01089v2). arXiv:1805.01089v2, IJCAI 2018.
28. Li Wang, Junlin Yao, Yunzhe Tao, Li Zhong, Wei Liu, Qiang Du. [A Reinforced Topic-Aware Convolutional Sequence-to-Sequence Model for Abstractive Text Summarization](https://arxiv.org/abs/1805.03616). arXiv:1805.03616, International Joint Conference on Artificial Intelligence and European Conference on Artificial Intelligence (IJCAI-ECAI), 2018.
29. Guokan Shang, Wensi Ding, Zekun Zhang, Antoine J.-P. Tixier, Polykarpos Meladianos, Michalis Vazirgiannis, Jean-Pierre Lorre´. [Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization](https://arxiv.org/abs/1805.05271). arXiv:1805.05271, 2018.
30. Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, Noah A. Smith. [Toward Abstractive Summarization Using Semantic Representations](https://arxiv.org/abs/1805.10399v1). arXiv:1805.10399, 2018.
30. Han Guo, Ramakanth Pasunuru, Mohit Bansal. [Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation](https://arxiv.org/abs/1805.11004v1). arXiv:1805.11004, ACL 2018.
29. Yen-Chun Chen, Mohit Bansal. [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](https://arxiv.org/abs/1805.11080). arXiv:1805.11080, ACL 2018. The souce code written in Python is [fast_abs_rl](https://github.com/chenrocks/fast_abs_rl).
30. Reinald Kim Amplayo, Seonjae Lim, Seung-won Hwang. [Entity Commonsense Representation for Neural Abstractive Summarization](https://arxiv.org/abs/1806.05504v1). arXiv:1806.05504, NAACL 2018.
30. Kaiqiang Song, Lin Zhao, Fei Liu. [Structure-Infused Copy Mechanisms for Abstractive Summarization](https://arxiv.org/abs/1806.05658v2). arXiv:1806.05658, 2018.
31. Kexin Liao, Logan Lebanoff, Fei Liu. [Abstract Meaning Representation for Multi-Document Summarization](https://arxiv.org/abs/1806.05655v1). arXiv:1806.05655, 2018.
30. Shibhansh Dohare, Vivek Gupta and Harish Karnick. [Unsupervised Semantic Abstractive Summarization](http://aclweb.org/anthology/P18-3011). ACL, July 2018.
31. Niantao Xie, Sujian Li, Huiling Ren, Qibin Zhai. [Abstractive Summarization Improved by WordNet-based Extractive Sentences](https://arxiv.org/abs/1808.01426v1). arXiv:1808.01426, NLPCC 2018.
31. Wojciech Kryściński, Romain Paulus, Caiming Xiong, Richard Socher. [Improving Abstraction in Text Summarization](https://arxiv.org/abs/1808.07913v1). arXiv:1808.07913, 2018.
31. Hardy, Andreas Vlachos. [Guided Neural Language Generation for Abstractive Summarization using Abstract Meaning Representation](https://arxiv.org/abs/1808.09160v1). arXiv:1808.09160, EMNLP 2018.
31. Sebastian Gehrmann, Yuntian Deng, Alexander M. Rush. [Bottom-Up Abstractive Summarization](https://arxiv.org/abs/1808.10792v1). arXiv:1808.10792, 2018.
31. Yichen Jiang, Mohit Bansal. [Closed-Book Training to Improve Summarization Encoder Memory](https://arxiv.org/abs/1809.04585v1). arXiv:1809.04585, 2018.
30. Raphael Schumann. [Unsupervised Abstractive Sentence Summarization using Length Controlled Variational Autoencoder](https://arxiv.org/abs/1809.05233). arXiv:1809.05233, 2018.
31. Kamal Al-Sabahi, Zhang Zuping, Yang Kang. [Bidirectional Attentional Encoder-Decoder Model and Bidirectional Beam Search for Abstractive Summarization](https://arxiv.org/abs/1809.06662v1). arXiv:1809.06662, 2018.
32. Tomonori Kodaira, Mamoru Komachi. [The Rule of Three: Abstractive Text Summarization in Three Bullet Points](https://arxiv.org/abs/1809.10867v1). arXiv:1809.10867, PACLIC 2018, 2018.
32. Byeongchang Kim, Hyunwoo Kim, Gunhee Kim. [Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://arxiv.org/abs/1811.00783). arXiv:1811.00783, 2018. The github project is  [MMN](https://github.com/ctr4si/MMN) including the dataset.


### Text Summarization

1. Eduard Hovy and Chin-Yew Lin. [Automated text summarization and the summarist system](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/tipster-proc-hovy-lin-final.pdf). In Proceedings of a Workshop on Held at Baltimore, Maryland, ACL, 1998.
2. Eduard Hovy and Chin-Yew Lin. [Automated Text Summarization in SUMMARIST](https://www.isi.edu/natural-language/people/hovy/papers/98hovylin-summarist.pdf). In Advances in Automatic Text Summarization, 1999.
3. Dipanjan Das and Andre F.T. Martins. [A survey on automatic text summarization](https://wtlab.um.ac.ir/images/e-library/text_summarization/A%20Survey%20on%20Automatic%20Text%20Summarization.pdf). Technical report, CMU, 2007
4. J. Leskovec, L. Backstrom, J. Kleinberg. [Meme-tracking and the Dynamics of the News Cycle](http://www.memetracker.org). ACM SIGKDD Intl. Conf. on Knowledge Discovery and Data Mining, 2009.
5. Ryang, Seonggi, and Takeshi Abekawa. "[Framework of automatic text summarization using reinforcement learning](http://dl.acm.org/citation.cfm?id=2390980)." In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 256-265. Association for Computational Linguistics, 2012. [not neural-based methods]
6. King, Ben, Rahul Jha, Tyler Johnson, Vaishnavi Sundararajan, and Clayton Scott. "[Experiments in Automatic Text Summarization Using Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8775&rep=rep1&type=pdf)." Machine Learning (2011).
7. Liu, Yan, Sheng-hua Zhong, and Wenjie Li. "[Query-Oriented Multi-Document Summarization via Unsupervised Deep Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5058/5322)." AAAI. 2012.
8. He, Zhanying, Chun Chen, Jiajun Bu, Can Wang, Lijun Zhang, Deng Cai, and Xiaofei He. "[Document Summarization Based on Data Reconstruction](http://cs.nju.edu.cn/zlj/pdf/AAAI-2012-He.pdf)." In AAAI. 2012.
9. Mohsen Pourvali, Mohammad Saniee Abadeh. [Automated Text Summarization Base on Lexicales Chain and graph Using of WordNet and Wikipedia Knowledge Base](https://arxiv.org/abs/1203.3586v1). arXiv:1203.3586, 2012.
10. PadmaPriya, G., and K. Duraiswamy. [An Approach For Text Summarization Using Deep Learning Algorithm](http://thescipub.com/PDF/jcssp.2014.1.9.pdf). Journal of Computer Science 10, no. 1 (2013): 1-9.
11. Rushdi Shams, M.M.A. Hashem, Afrina Hossain, Suraiya Rumana Akter, Monika Gope. [Corpus-based Web Document Summarization using Statistical and Linguistic Approach](https://arxiv.org/abs/1304.2476v1). arXiv:1304.2476, Procs. of the IEEE International Conference on Computer and Communication Engineering (ICCCE10), pp. 115-120, Kuala Lumpur, Malaysia, May 11-13, (2010).
12. Juan-Manuel Torres-Moreno. [Beyond Stemming and Lemmatization: Ultra-stemming to Improve Automatic Text Summarization](https://arxiv.org/abs/1209.3126). arXiv:1209.3126, 2012.
13. Rioux, Cody, Sadid A. Hasan, and Yllias Chali. [Fear the REAPER: A System for Automatic Multi-Document Summarization with Reinforcement Learning](http://emnlp2014.org/papers/pdf/EMNLP2014075.pdf). In EMNLP, pp. 681-690. 2014.[not neural-based methods]
14. Fatma El-Ghannam, Tarek El-Shishtawy. [Multi-Topic Multi-Document Summarizer](https://arxiv.org/abs/1401.0640v1). arXiv:1401.0640, 2014.
15. Denil, Misha, Alban Demiraj, and Nando de Freitas. [Extraction of Salient Sentences from Labelled Documents](http://arxiv.org/abs/1412.6815). arXiv:1412.6815,  2014.
16. Denil, Misha, Alban Demiraj, Nal Kalchbrenner, Phil Blunsom, and Nando de Freitas.[Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network](http://arxiv.org/abs/1406.3830). arXiv:1406.3830, 2014.
17. Cao, Ziqiang, Furu Wei, Li Dong, Sujian Li, and Ming Zhou. [Ranking with Recursive Neural Networks and Its Application to Multi-document Summarization](http://gana.nlsde.buaa.edu.cn/~lidong/aaai15-rec_sentence_ranking.pdf). AAAI, 2015.
18. Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, and Noah A. Smith. [Toward Abstractive Summarization Using Semantic Representations](http://www.cs.cmu.edu/~nasmith/papers/liu+flanigan+thomson+sadeh+smith.naacl15.pdf). NAACL, 2015.
19. Wenpeng Yin， Yulong Pei. Optimizing Sentence Modeling and Selection for Document Summarization. IJCAI, 2015.
20. Liu, He, Hongliang Yu, and Zhi-Hong Deng. [Multi-Document Summarization Based on Two-Level Sparse Representation Model](http://www.cis.pku.edu.cn/faculty/system/dengzhihong/papers/AAAI%202015_Multi-Document%20Summarization%20Based%20on%20Two-Level%20Sparse%20Representation%20Model.pdf). In Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
21. Jin-ge Yao, Xiaojun Wan and Jianguo Xiao. [Compressive Document Summarization via Sparse Optimization](http://ijcai.org/Proceedings/15/Papers/198.pdf). IJCAI, 2015.
22. Piji Li, Lidong Bing, Wai Lam, Hang Li, and Yi Liao. [Reader-Aware Multi-Document Summarization via Sparse Coding](http://arxiv.org/abs/1504.07324). arXiv:1504.07324,  IJCAI, 2015.
23. Marta Aparício, Paulo Figueiredo, Francisco Raposo, David Martins de Matos, Ricardo Ribeiro, Luís Marujo. [Summarization of Films and Documentaries Based on Subtitles and Scripts](https://arxiv.org/abs/1506.01273v3). arXiv:1506.01273, 2015.
24. Luís Marujo, Ricardo Ribeiro, David Martins de Matos, João P. Neto, Anatole Gershman, Jaime Carbonell. [Extending a Single-Document Summarizer to Multi-Document: a Hierarchical Approach](https://arxiv.org/abs/1507.02907v1). arXiv:1507.02907, 2015.
25. Xiaojun Wan, Yansong Feng and Weiwei Sun. [Automatic Text Generation: Research Progress and Future Trends](http://www.icst.pku.edu.cn/lcwm/wanxj/files/TextGenerationSurvey.pdf). Book Chapter in CCF 2014-2015 Annual Report on Computer Science and Technology in China (In Chinese), 2015.
26. Xiaojun Wan, Ziqiang Cao, Furu Wei, Sujian Li, Ming Zhou. [Multi-Document Summarization via Discriminative Summary Reranking](https://arxiv.org/abs/1507.02062v1).     arXiv:1507.02062, 2015.
26. Gulcehre, Caglar, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. [Pointing the Unknown Words](http://arxiv.org/abs/1603.08148). arXiv:1603.08148, 2016.
27. Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393). arXiv:1603.06393, ACL, 2016.
    - They addressed an important problem in sequence-to-sequence (Seq2Seq) learning referred to as copying, in which certain segments in the input sequence are selectively replicated in the output sequence. In this paper, they incorporated copying into neural network-based Seq2Seq learning and propose a new model called CopyNet with encoder-decoder structure. CopyNet can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose sub-sequences in the input sequence and put them at proper places in the output sequence.
28. Jianmin Zhang, Jin-ge Yao and Xiaojun Wan. [Toward constructing sports news from live text commentary](http://www.icst.pku.edu.cn/lcwm/wanxj/files/acl16_sports.pdf). In Proceedings of ACL, 2016.
29. Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. "[AttSum: Joint Learning of Focusing and Summarization with Neural Attention](http://arxiv.org/abs/1604.00125)".  arXiv:1604.00125, 2016
30. Ayana, Shiqi Shen, Yu Zhao, Zhiyuan Liu and Maosong Sun. [Neural Headline Generation with Sentence-wise Optimization](https://arxiv.org/abs/1604.01904). arXiv:1604.01904, 2016.
31. Ayana, Shiqi Shen, Zhiyuan Liu and Maosong Sun. [Neural Headline Generation with Minimum Risk Training](https://128.84.21.199/abs/1604.01904v1). 2016.
32. Lu Wang, Hema Raghavan, Vittorio Castelli, Radu Florian, Claire Cardie. [A Sentence Compression Based Framework to Query-Focused Multi-Document Summarization](https://arxiv.org/abs/1606.07548v1). arXiv:1606.07548, 2016.
33. Milad Moradi, Nasser Ghadiri. [Different approaches for identifying important concepts in probabilistic biomedical text summarization](https://arxiv.org/abs/1605.02948v3). arXiv:1605.02948, 2016.
34. Kikuchi, Yuta, Graham Neubig, Ryohei Sasano, Hiroya Takamura, and Manabu Okumura. [Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552). arXiv:1609.09552, 2016.
8. Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei and Hui Jiang. [Distraction-Based Neural Networks for Document Summarization](https://arxiv.org/abs/1610.08462). arXiv:1610.08462, IJCAI, 2016.
35. Wang, Lu, and Wang Ling. [Neural Network-Based Abstract Generation for Opinions and Arguments](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf). NAACL, 2016.
36. Yishu Miao, Phil Blunsom.  [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](http://arxiv.org/abs/1609.07317).  EMNLP, 2016.
37. Takase, Sho, Jun Suzuki, Naoaki Okazaki, Tsutomu Hirao, and Masaaki Nagata. [Neural headline generation on abstract meaning representation](https://www.aclweb.org/anthology/D/D16/D16-1112.pdf).  EMNLP, 1054-1059, 2016.
38. Wenyuan Zeng, Wenjie Luo, Sanja Fidler, Raquel Urtasun.  [Efficient Summarization with Read-Again and Copy Mechanism](https://arxiv.org/abs/1611.03382). arXiv:1611.03382, 2016.
39. Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. [Improving Multi-Document Summarization via Text Classification](https://arxiv.org/abs/1611.09238v1). arXiv:1611.09238, 2016.
40. Hongya Song, Zhaochun Ren, Piji Li, Shangsong Liang, Jun Ma, and Maarten de Rijke. [Summarizing Answers in Non-Factoid Community Question-Answering](http://dl.acm.org/citation.cfm?id=3018704). In WSDM 2017: The 10th International Conference on Web Search and Data Mining, 2017.
41. Piji Li, Zihao Wang, Wai Lam, Zhaochun Ren, Lidong Bing.  [Salience Estimation via Variational Auto-Encoders for Multi-Document Summarization](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14613). In AAAI, 2017.
42. Yinfei Yang, Forrest Sheng Bao, Ani Nenkova. [Detecting (Un)Important Content for Single-Document News Summarization](https://arxiv.org/abs/1702.07998v1). arXiv:1702.07998, 2017.
43. Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky, Yu Chi. [Deep Keyphrase Generation](https://arxiv.org/abs/1704.06879v1). arXiv:1704.06879, 2017. The source code written in Python is [seq2seq-keyphrase](https://github.com/memray/seq2seq-keyphrase).
44. Abigail See, Peter J. Liu and Christopher D. Manning. [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368). ACL, 2017. The souce code is [pointer-generator](https://github.com/abisee/pointer-generator).
45. Qingyu Zhou, Nan Yang, Furu Wei and Ming Zhou. [Selective Encoding for Abstractive Sentence Summarization](https://arxiv.org/abs/1704.07073). arXiv:1704.07073, ACL, 2017.
46. Maxime Peyrard and Judith Eckle-Kohler. [Supervised Learning of Automatic Pyramid for Optimization-Based Multi-Document Summarization](). ACL, 2017.
47. Jin-ge Yao, Xiaojun Wan and Jianguo Xiao. [Recent Advances in Document Summarization](http://www.icst.pku.edu.cn/lcwm/wanxj/files/summ_survey_draft.pdf). KAIS, survey paper, 2017.
48. Pranay Mathur, Aman Gill and Aayush Yadav. [Text Summarization in Python: Extractive vs. Abstractive techniques revisited](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/#text_summarization_in_python). 2017.
    - They compared modern extractive methods like LexRank, LSA, Luhn and Gensim’s existing TextRank summarization module on the [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) of 51 (article, summary) pairs. They also had a try with an abstractive technique using Tensorflow’s algorithm [textsum](https://github.com/tensorflow/models/tree/master/textsum), but didn’t obtain good results due to its extremely high hardware demands (7000 GPU hours).
49. Arman Cohan, Nazli Goharian. [Scientific Article Summarization Using Citation-Context and Article's Discourse Structure](https://arxiv.org/abs/1704.06619v1). arXiv:1704.06619, EMNLP, 2015.
50. Shuming Ma, Xu Sun, Jingjing Xu, Houfeng Wang, Wenjie Li, Qi Su. [Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization](https://arxiv.org/abs/1706.02459). The source code written in Python is [SRB](https://github.com/lancopku/SRB).
50. Arman Cohan, Nazli Goharian. [Scientific document summarization via citation contextualization and scientific discourse](https://arxiv.org/abs/1706.03449v1). arXiv:1706.03449, 2017.
51. Michihiro Yasunaga, Rui Zhang, Kshitijh Meelu, Ayush Pareek, Krishnan Srinivasan, Dragomir Radev. [Graph-based Neural Multi-Document Summarization](https://arxiv.org/abs/1706.06681v3). arXiv:1706.06681, CoNLL, 2017.
52. Abeed Sarker, Diego Molla, Cecile Paris. [Automated text summarisation and evidence-based medicine: A survey of two domains](https://arxiv.org/abs/1706.08162v1). arXiv:1706.08162, 2017.
53. Mehdi Allahyari, Seyedamin Pouriyeh, Mehdi Assefi, Saeid Safaei, Elizabeth D. Trippe, Juan B. Gutierrez, Krys Kochut. [Text Summarization Techniques: A Brief Survey](https://arxiv.org/abs/1707.02268). arXiv:1707.02268, 2017.
5. Demian Gholipour Ghalandari. [Revisiting the Centroid-based Method: A Strong Baseline for Multi-Document Summarization](https://arxiv.org/abs/1708.07690v1). arXiv:1708.07690, EMNLP, 2017.
54. Shuming Ma, Xu Sun. [A Semantic Relevance Based Neural Network for Text Summarization and Text Simplification](https://arxiv.org/abs/1710.02318v1). arXiv:1710.02318, 2017. The source code written in Python is [SRB](https://github.com/lancopku/SRB).
55. Kaustubh Mani, Ishan Verma, Lipika Dey. [Multi-Document Summarization using Distributed Bag-of-Words Model](https://arxiv.org/abs/1710.02745v1). arXiv:1710.02745, 2017.
56. Liqun Shao, Hao Zhang, Ming Jia, Jie Wang. [Efficient and Effective Single-Document Summarizations and A Word-Embedding Measurement of Quality](https://arxiv.org/abs/1710.00284v1). arXiv:1710.00284, KDIR, 2017.
57. Mohammad Ebrahim Khademi, Mohammad Fakhredanesh, Seyed Mojtaba Hoseini. [Conceptual Text Summarizer: A new model in continuous vector space](https://arxiv.org/abs/1710.10994v2). arXiv:1710.10994, 2017.
58. Jingjing Xu. [Improving Social Media Text Summarization by Learning Sentence Weight Distribution](https://arxiv.org/abs/1710.11332v1). arXiv:1710.11332, 2017.
59. Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, Noam Shazeer. [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198v1). arXiv:1801.10198, 2018.
60. Parth Mehta, Prasenjit Majumder. [Content based Weighted Consensus Summarization](https://arxiv.org/abs/1802.00946v1). arXiv:1802.00946, 2018.
61. Mayank Chaudhari, Aakash Nelson Mattukoyya. [Tone Biased MMR Text Summarization](https://arxiv.org/abs/1802.09426v2). arXiv:1802.09426, 2018.
62. Divyanshu Daiya, Anukarsh Singh, Mukesh Jadon. [Using Statistical and Semantic Models for Multi-Document Summarization](https://arxiv.org/abs/1805.04579v2). arXiv:1805.04579, 2018.
62. Wan-Ting Hsu, Chieh-Kai Lin, Ming-Ying Lee, Kerui Min, Jing Tang, Min Sun. [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](https://arxiv.org/abs/1805.06266v2). arXiv:1805.06266, ACL 2018.
62. Pei Guo, Connor Anderson, Kolten Pearson, Ryan Farrell. [Neural Network Interpretation via Fine Grained Textual Summarization](https://arxiv.org/abs/1805.08969v2). arXiv:1805.08969, 2018.
63. Kamal Al-Sabahi, Zhang Zuping, Yang Kang. [Latent Semantic Analysis Approach for Document Summarization Based on Word Embeddings](https://arxiv.org/abs/1807.02748v1). arXiv:1807.02748, KSII Transactions on Internet and Information Systems, 2018.
63. Chandra Khatri, Gyanit Singh, Nish Parikh. [Abstractive and Extractive Text Summarization using Document Context Vector and Recurrent Neural Networks](https://arxiv.org/abs/1807.08000v2). arXiv:1807.08000v2, ACM KDD 2018 Deep Learning Day, 2018.
63. Logan Lebanoff, Kaiqiang Song, Fei Liu. [Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization](https://arxiv.org/abs/1808.06218v2). arXiv:1808.06218, 2018.
64. Shashi Narayan, Shay B. Cohen, Mirella Lapata. [Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745v1). arXiv:1808.08745, 2018.
62. Parth Mehta, Prasenjit Majumder. [Exploiting local and global performance of candidate systems for aggregation of summarization techniques](https://arxiv.org/abs/1809.02343v1). arXiv:1809.02343, 2018.
63. Chi Zhang, Shagan Sah, Thang Nguyen, Dheeraj Peri, Alexander Loui, Carl Salvaggio, Raymond Ptucha. [Semantic Sentence Embeddings for Paraphrasing and Text Summarization](https://arxiv.org/abs/1809.10267v1). arXiv:1809.10267, IEEE GlobalSIP 2017 Conference, 2018.
64. Yaser Keneshloo, Naren Ramakrishnan, Chandan K. Reddy. [Deep Transfer Reinforcement Learning for Text Summarization](https://arxiv.org/abs/1810.06667v1). arXiv:1810.06667, 2018.


### Chinese Text Summarization

1. Mao Song Sun. [Natural Language Processing Based on Naturally Annotated Web Resources](http://www.thunlp.org/site2/images/stories/files/2011_zhongwenxinxixuebao_sms.pdf). Journal of Chinese Information Processing, 2011.
2. Baotian Hu, Qingcai Chen and Fangze Zhu. [LCSTS: A Large Scale Chinese Short Text Summarization Dataset](https://arxiv.org/abs/1506.05865). 2015.
   - They constructed a large-scale Chinese short text summarization dataset constructed from the Chinese microblogging website Sina Weibo, which is released to [the public](http://icrc.hitsz.edu.cn/Article/show/139.html). Then they performed GRU-based encoder-decoder method on it to generate summary. They took the whole short text as one sequence, this may not be very reasonable, because most of short texts contain several sentences.
   - LCSTS contains 2,400,591 (short text, summary) pairs as the training set and 1,106  pairs as the test set.
   - All the models are trained on the GPUs tesla M2090 for about one week.
   - The results show that the RNN with context outperforms RNN without context on both character and word based input.
   - Moreover, the performances of the character-based input outperform the word-based input.
3. Bingzhen Wei, Xuancheng Ren, Xu Sun, Yi Zhang, Xiaoyan Cai, Qi Su. [Regularizing Output Distribution of Abstractive Chinese Social Media Text Summarization for Improved Semantic Consistency](https://arxiv.org/abs/1805.04033v1). arXiv:1805.04033, 2018.
4. [LancoSum](https://github.com/lancopku/LancoSum) provides a toolkit for abstractive summarization, which can achieve the SOTA performance.
   * Shuming Ma, Xu Sun, Wei Li, Sujian Li, Wenjie Li, Xuancheng Ren. [Query and Output: Generating Words by Querying Distributed Word Representations for Paraphrase Generation](https://arxiv.org/abs/1803.01465v3). arXiv:1803.01465v3, NAACL HLT 2018.
   * Junyang Lin, Xu Sun, Shuming Ma, Qi Su. [Global Encoding for Abstractive Summarization](https://arxiv.org/abs/1805.03989v2). arXiv:1805.03989v2, ACL 2018. The source code written in Python is [Global-Encoding](https://github.com/lancopku/Global-Encoding).
   * Shuming Ma, Xu Sun, Junyang Lin and Houfeng Wang. [Autoencoder as Assistant Supervisor: Improving Text Representation for Chinese Social Media Text Summarization](https://arxiv.org/abs/1805.04869v1). arXiv:1805.04869, ACL 2018. The source code written in Python is [superAE](https://github.com/lancopku/superAE).

### Evaluation Metrics

1. Chin-Yew Lin and Eduard Hovy. [Automatic Evaluation of Summaries Using N-gram
Co-Occurrence Statistics](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/naacl2003.pdf). In Proceedings of the Human Technology Conference 2003 (HLT-NAACL-2003).
2. Chin-Yew Lin. [Rouge: A package for automatic evaluation of summaries](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/was2004.pdf). Workshop on Text Summarization Branches Out, Post-Conference Workshop of ACL 2004.
3. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).
4. Arman Cohan, Nazli Goharian. [Revisiting Summarization Evaluation for Scientific Articles](https://arxiv.org/abs/1604.00400v1). arXiv:1604.00400, LREC, 2016.
5. Maxime Peyrard. [A Formal Definition of Importance for Summarization](https://arxiv.org/abs/1801.08991v1). arXiv:1801.08991, 2018.
6. Kavita Ganesan. [ROUGE 2.0: Updated and Improved Measures for Evaluation of Summarization Tasks](https://arxiv.org/abs/1803.01937v1). arXiv:1803.01937, 2018. It works by comparing an automatically produced summary or translation against a set of reference summaries (typically human-produced).  ROUGE is one of the standard ways to compute effectiveness of auto generated summaries. The evaluation toolkit [ROUGE 2.0](https://github.com/RxNLP/ROUGE-2.0) is an easy to use  for Automatic Summarization tasks. 

### Opinion Summarization

1. Kavita Ganesan, ChengXiang Zhai and Jiawei Han. [Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions](http://kavita-ganesan.com/opinosis). Proceedings of COLING '10, 2010.
2. Kavita Ganesan, ChengXiang Zhai and Evelyne Viegas. [Micropinion Generation: An Unsupervised Approach to Generating Ultra-Concise Summaries of Opinions](http://kavita-ganesan.com/micropinion-generation). WWW'12, 2012.
3. Kavita Ganesan. [Opinion Driven Decision Support System (ODSS)](http://kavita-ganesan.com/phd-thesis). PhD Thesis, University of Illinois at Urbana-Champaign, 2013.
4. Ozan Irsoy and Claire Cardie. [Opinion Mining with Deep Recurrent Neural Networks](https://www.cs.cornell.edu/~oirsoy/files/emnlp14drnt.pdf). In EMNLP, 2014.
5. Ahmad Kamal. [Review Mining for Feature Based Opinion Summarization and Visualization](https://arxiv.org/abs/1504.03068v2). arXiv:1504.03068, 2015.
6. Haibing Wu, Yiwei Gu, Shangdi Sun and Xiaodong Gu. [Aspect-based Opinion Summarization with Convolutional Neural Networks](https://arxiv.org/abs/1511.09128). 2015.
7. Lu Wang, Hema Raghavan, Claire Cardie, Vittorio Castelli. [Query-Focused Opinion Summarization for User-Generated Content](https://arxiv.org/abs/1606.05702v1). arXiv:1606.05702, 2016.

