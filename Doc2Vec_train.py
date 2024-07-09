import gensim
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora import WikiCorpus
from tqdm import tqdm


def train_doc2vec(vector_size=32):
    # 指定所需的句向量维度和其他参数
    window = 5
    min_count = 5
    workers = multiprocessing.cpu_count()

    print('训练Doc2Vec模型')
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=20)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    print('保存训练好的Doc2Vec模型')
    model.save(model_save_path + f"pretrained_wiki_doc2vec_{vector_size}.model")

    # 输入新的句子并生成句向量
    new_sentence = "This is a new sentence for embedding."
    new_sentence_vector = model.infer_vector(new_sentence.split())

    # 打印生成的句向量
    print("测试")
    print(new_sentence_vector)
    print(len(new_sentence_vector))


if __name__ == '__main__':
    model_save_path = "./predata/wiki_doc2vec/"

    # 下载Wikipedia的XML数据（例如，英文维基百科）
    # 数据下载地址：https://dumps.wikimedia.org/enwiki/latest/   选择enwiki-latest-pages-articles.xml.bz2等
    print('将Wikipedia XML数据转换为纯文本语料库')
    wiki_dump_path = "./enwiki-latest-pages-articles1.xml-p1p41242.bz2"
    wiki_corpus = WikiCorpus(wiki_dump_path)
    # 预处理并标记化Wikipedia语料库中的文档
    tagged_data = []
    for i, text in tqdm(enumerate(wiki_corpus.get_texts()), desc='预处理wiki语料库'):
        tagged_data.append(TaggedDocument(words=text, tags=[str(i)]))

    train_doc2vec(32)
    train_doc2vec(64)
    train_doc2vec(96)
    train_doc2vec(128)
