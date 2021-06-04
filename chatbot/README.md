* **ChatBot分析**  
  1.包含語意分析(NLP,NLU)  
  2.WebAPI(使用FlaskAPI撰寫, ChatBot設定介面所使用的)  
  
* **檔案說明**  
FlaskApi_HelperIntentModel_Train.py : flask撰寫的API , 用於模型訓練   
FlaskApi_Check_HelperIntentModel.py : 檢查API參數的decorators  
NLP_IntentPreprocessing.py : 前處理通用程式(結巴斷詞, 資料前處理......)  
NLP_IntentModelFactory.py : 模型建立工廠  
NLP_IntentDLModel.py : 深度學習模型(使用Keras)  
NLP_IntentMLModel.py : 機器學習模型(使用scikit-learn)  
NLP_IntentBertModel.py : Bert模型(使用pytorch-pretrained-bert)  
notebook_ML_intent_training.ipynb : 訓練模型實驗程式    

<img src="github_demo_flow2.jpg" height="400" width="800">
