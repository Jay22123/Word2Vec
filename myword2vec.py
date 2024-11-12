from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import xml.etree.ElementTree as ET
from gensim.parsing.preprocessing import STOPWORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
from adjustText import adjust_text
from sklearn.preprocessing import MinMaxScaler


class Function():

    def get_text_without_tags(self, element):
        parts = [element.text or '']
        for subelem in element:
            parts.append(self.get_text_without_tags(subelem))
            if subelem.tail:
                parts.append(' ' + subelem.tail.strip() + ' ')
            else:
                parts.append(' ')
        return ''.join(parts).strip()

    def load_data(self, filepath):
        abstract_texts = []
        content = ET.iterparse(filepath, events=('start', 'end'))
        for event, elem in content:
            if event == 'end' and elem.tag == 'content':
                text = self.get_text_without_tags(elem)
                abstract_texts.append(text)
                elem.clear()

        combined_text = ' '.join(abstract_texts)
        content = combined_text

        sentences = []
        # 使用 simple_preprocess 進行分詞
        tokens = [word for word in simple_preprocess(
            content) if word not in STOPWORDS]
        sentences.append(tokens)

        # 計算詞頻
        self.word_counts = Counter(tokens)

        

        self.sentenes = sentences

        

    def Normal_Train(self, sentences):
        model = Word2Vec(sentences, vector_size=50,
                         window=5, min_count=1, workers=4)
        # 保存模型
        model.save("word2vec.model")
        print("Word2Vec 模型訓練完成！")

    def Find_Best_Train(self, sentences):
        dimensions = [50, 100, 200, 300]
        # 假設我們有一組詞對和其預期的相似性標籤 (1 表示相似，0 表示不相似)
        word_pairs = [("covid", "sleep"), ("patient", "disease")]
        expected_similarity = [1, 1]  # 預期結果：這些詞應該是相似的
        results = []
        for dim in dimensions:
            model = Word2Vec(sentences, vector_size=dim,
                             window=5, min_count=1, workers=4)
            similarities = []

            # 計算每對詞的相似度
            for word1, word2 in word_pairs:
                if word1 in model.wv and word2 in model.wv:
                    similarity = model.wv.similarity(word1, word2)
                    # 使用 0.5 作為相似度閾值（大於 0.5 表示相似）
                    similarities.append(1 if similarity > 0.5 else 0)
                else:
                    similarities.append(0)  # 如果詞不存在，標記為不相似

                # 計算準確率
            accuracy = accuracy_score(expected_similarity, similarities)
            results.append((dim, accuracy))
        # 找出最佳維度
        best_dimension = max(results, key=lambda x: x[1])
        print("最佳維度:", best_dimension[0], "準確率:", best_dimension[1])

        # 保存模型
        model.save("word2ve_Best.model")
        print("Word2Vec 模型訓練完成！")

    def LoadNormalModel(self, NormalModel):
        self.Normalmodel = Word2Vec.load(NormalModel)
        

    def LoadBestModel(self, BestModel):
        self.Bestmodel = Word2Vec.load(BestModel)
       

    def GetNormalWordCloud(self, target_word, word_numbers):
        # 取得與特定詞相似的詞及其相似度

        similar_words = self.Normalmodel.wv.most_similar(
            target_word, topn=word_numbers)
        word_freq = {word: similarity for word, similarity in similar_words}
        # 生成詞雲
        wordcloud = WordCloud(
            width=800, height=400).generate_from_frequencies(word_freq)
        # 保存詞雲圖片
        wordcloud.to_file("static/images/Normal_wordcloud.png")



    def GetBestWordCloud(self, target_word, word_numbers):
        # 取得與特定詞相似的詞及其相似度

        similar_words2 = self.Bestmodel.wv.most_similar(
            target_word, topn=word_numbers)
        word_freq2 = {word: similarity for word, similarity in similar_words2}
        # 生成詞雲
        wordcloud2 = WordCloud(
            width=800, height=400).generate_from_frequencies(word_freq2)
        # 保存詞雲圖片
        wordcloud2.to_file("static/images/Best_wordcloud.png")


    def Get_Countwords(self,range):
        # 找出高頻字的前 50 名
        position = int((int(range)/100) * len(self.word_counts))
        start = max(0, position - 25)  # 確保起始位置不低於 0
        end = min(len(self.word_counts), position + 25)  # 確保結束位置不超出列表長度

        # 將 Counter 轉換為排序列表，並進行切片
        sorted_words = self.word_counts.most_common()  # 排序後的高頻詞列表
        selected_words = sorted_words[start:end]  # 提取指定範圍的詞
        return selected_words



    def GetNormalModelWords(self):
        print(len(self.Normalmodel.wv.index_to_key))
        return self.Normalmodel.wv.index_to_key[:200]

    def GetBestModelWords(self):
       return self.Bestmodel.wv.index_to_key[:200] 

    def GetNormalModel2D(self):
        
        words = self.Normalmodel.wv.index_to_key[:200] 
        word_vectors = [self.Normalmodel.wv[word] for word in words]

        # 使用 PCA 將詞向量降維到 2 維
        pca = PCA(n_components=2)
        result = pca.fit_transform(word_vectors)

        # 使用 MinMaxScaler 將 X 和 Y 坐標縮放到指定範圍
        scaler = MinMaxScaler(feature_range=(-5, 5))  # 將範圍設為 (-5, 5) 更分散
        scaled_result = scaler.fit_transform(result)

        # 繪製結果
        plt.figure(figsize=(15, 15))
        texts = []
        for i, word in enumerate(words):
            plt.scatter(scaled_result[i, 0], scaled_result[i, 1])
            texts.append(plt.text(scaled_result[i, 0], scaled_result[i, 1], word, fontsize=12))

        # 使用 adjust_text 調整文字位置，避免重疊，並允許自由移動
        adjust_text(texts, only_move={'points': 'xy', 'text': 'xy'},
                    arrowprops=dict(arrowstyle='->', color='gray'))

        # 設置軸範圍
        plt.xlim(-5.5, 5.5)
        plt.ylim(-5.5, 5.5)

        # 保存圖片
        plt.title("PCA of Word2Vec Word Vectors")
        plt.savefig("static/images/Normal_2D.png")
       

    def GetBestModel2D(self):
        words = self.Bestmodel.wv.index_to_key[:200]  
        word_vectors = [self.Bestmodel.wv[word] for word in words]

        # 使用 PCA 將詞向量降維到 2 維
        pca = PCA(n_components=2)
        result = pca.fit_transform(word_vectors)

        # 使用 MinMaxScaler 將 X 和 Y 坐標縮放到指定範圍
        scaler = MinMaxScaler(feature_range=(-5, 5))  # 將範圍設為 (-5, 5) 更分散
        scaled_result = scaler.fit_transform(result)

        # 繪製結果
        plt.figure(figsize=(15, 15))
        texts = []
        for i, word in enumerate(words):
            plt.scatter(scaled_result[i, 0], scaled_result[i, 1])
            texts.append(plt.text(scaled_result[i, 0], scaled_result[i, 1], word, fontsize=12))

        # 使用 adjust_text 調整文字位置，避免重疊，並允許自由移動
        adjust_text(texts, only_move={'points': 'xy', 'text': 'xy'},
                    arrowprops=dict(arrowstyle='->', color='gray'))

        # 設置軸範圍
        plt.xlim(-5.5, 5.5)
        plt.ylim(-5.5, 5.5)

        # 保存圖片
        plt.title("PCA of Word2Vec Word Vectors")
        plt.savefig("static/images/Best_2D.png")


    def calNormalVec(self,word1,word2,word3):
        # 計算向量運算：word1 - word2 + word3
        result_vector = self.Normalmodel.wv[word1] - self.Normalmodel.wv[word2] + self.Normalmodel.wv[word3]

        # 找到最接近 result_vector 的詞
        similar_words = self.Normalmodel.wv.similar_by_vector(result_vector, topn=1)  # 找出最接近的詞
        return similar_words[0][0]

    def calBestVec(self,word1,word2,word3):
        # 計算向量運算：word1 - word2 + word3
        result_vector = self.Bestmodel.wv[word1] - self.Bestmodel.wv[word2] + self.Bestmodel.wv[word3]

        # 找到最接近 result_vector 的詞
        similar_words = self.Bestmodel.wv.similar_by_vector(result_vector, topn=1)  # 找出最接近的詞

        return similar_words[0][0]
        
