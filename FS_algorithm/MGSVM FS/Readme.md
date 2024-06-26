# MGSVM FS 
Multiclass group feature selection algorithm based on an all-in-one support vector machine

**Function**: 

Tackle the feature selection problem for structured data and multiclass classification tasks.

**Characteristics**: 
* Embedded feature selection
* Group feature selection / Group LASSO for structured data / Structural sparsity
* Multiclass classification / All-in-one SVM


## Usage
1. Input your own dataset **(X,y)**
2. Set the target number **Nr_FG_target** of feature groups that you want to select 
3. Adjust the searching vector **C_vec** for the trade-off parameter **C** to improve the efficiency, if necessary
4. Implement the following command

        python Example_MGSVM_FS.py


## Example results
### Input dataset
The MM features of the four material classes for the modality combination R+d+DoLP with labels of material class. 

### Target at selecting three spectral channels (feature groups)
Nr_WL_target = 3     
Nr_FG_target = Nr_WL_target
### The map of absolute values for all coefficients corresponding to different features
![alt text](Fig_Abs_Coeff__R+d+DoLP_SC2_Nr_SCs_3.png)
Absolute values of the coefficients represent the importance of spectral channels and modalities in distinguishing different material classes.

### The selected spectral channels according to the threshold 0.001 of absolute coefficient values
![alt text](Fig_Selected_Coeff__R+d+DoLP_SC2_Nr_SCs_3.png)


