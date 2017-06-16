import pandas as pd
import numpy as np

ACTION201602_FILE ='./datasets/JData_Action_201602.csv'
ACTION201603_FILE ='./datasets/JData_Action_201603.csv'
ACTION201604_FILE ='./datasets/JData_Action_201604.csv'
ACTION_FILE='./datasets/JData_Action.csv'


if __name__ == '__main__':
    print("开始拼接文件.")
    JData_201602 = pd.read_csv(ACTION201602_FILE)
    JData_201603 = pd.read_csv(ACTION201603_FILE)
    JData_201604 = pd.read_csv(ACTION201604_FILE)
    JData = pd.concat([JData_201602, JData_201603, JData_201604],axis=0)
    JData.to_csv('./datasets/JData_Action.csv',index=False)
    print("保存拼接文件结果.")