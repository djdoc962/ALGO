# Introduction
從已得到匹配(matches)的資訊後，根據此匹配資訊找變換函數(transformation)，使得匹配的對應關係(Correspondence)可以成立。

# Algorithms

1. RANSAC(Random sample consensus):
   一種用來估測模型參數的迭代演算法，可以`免疫(無視)於觀察數據中的離群點(outliers)`，即使很多離群點出現也能找到最優的參數模型可擬和全部`內群點(inliers)`。
   <img src="./assets/RANSAC1.jpg" width="300" />

   演算法基本流程：
   每次迭代會找出一個參數模型來擬合隨機選取的n個數據，將其他未選取的數據若丟入目前的參數模型所得的`誤差值(error)若 < 臨界值(Threshold value)`，則先被收藏為inliers，接著統計`inliers總數目若超過某個臨界值則更新參數模型來擬合當初隨機選取的n數據和新收藏的inliers`，重複前述過程直到迭代次數，即產生一個最佳參數模型。
   影像對準實驗則是預設套用RANSAC來找出影像的變換函數(transformation)，換句話說就是要計算出兩張影像之間的最佳的Homography matrix(通常用H表示)，必要的資訊為對應的`關鍵點座標，而且至少需要4對關鍵點座標`，步驟就如前述基本流程迭代出最佳的H矩陣。

   **缺點**
   演算法中的參數像是迭代次數，誤差值和臨界值的設定可能影響找到的Homography正確性。

   

