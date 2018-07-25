# 説明書  
  
  
  
まず、データを準備してください。  
data.pyのDataのクラースのオブジェクトを作ると、  
自動的、newLogsとoldLogsのリポジトリを読んでデータを準備される。  

> from data import Data  
> data = Data()  
  
そして、ニューラルネットワークを作ってください。  
  
> from nn import NeuralNetwork  
> neuralNetwork = NeuralNetwork(data.inputLen(), data.outputLen(), hiddenLayers = [40,40])  
  
hiddenLayersのパラメーターは隠れ層の形になります。  
ここで、40個のニューロンの隠れ層を2枚作ります。  
  
学習するとき、学習のメソッドは4つ要ります。  
data : データのオブジェクト  
epoch : 何回学習するか  
regularization : 正則化  
learningRate : 学習率  
  
> neuralNetwork.train(data, epoch=5000, regularization=0.03, learningRate = 0.01)  
  
学習したら、Plotでテストセットを使ってニューラルネットワークを評価して図表を作れます。  
  
> from plot import Plot  
> plot = Plot(data, neuralNetwork)  
> plot.plotResultBarChart()  
  
新しいログを分析することめできます。  
  
> neuralNetwork.printResult(data, "テストです")  
  
学習したモデルを保存できます。  
  
> neuralNetwork.save("/tmp/backup")  
  
保存したモデルに戻れます。  
  
> neuralNetwork.restore("/tmp/backup")  