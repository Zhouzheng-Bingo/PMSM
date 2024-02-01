import numpy as np



if __name__ == '__main__':
    #preds = np.load('./predictions.npy',allow_pickle=True)
    preds = np.load('predictions.npy')
    pres = np.load('y_test.npy')
    i=0
    a = np.array([])
    b = np.array([])
    
    for row_idx in range(preds.shape[0]): # 遍历行
        for col_idx in range(preds.shape[1]): # 遍历列

                # 获取当前值
            pred_value = preds[row_idx, col_idx]
            pres_value = pres[row_idx, col_idx]
            #print(pred_value)
            #print(pres_value)
        
        
        if (i%10000) == 0:
            #print(preds[row_idx])
            a = np.append(a,preds[row_idx])
            b = np.append(b,pres[row_idx])
            
            
        i = i + 1
    preds_down=a.reshape(200,4)
    pres_down=b.reshape(200,4)
    np.save("predictions_down.npy",preds_down)
    np.save("y_test_down.npy", pres_down)
    