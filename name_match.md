## Name Match Test Result

#### To match company name bwteen Dataset SR and BG, I use two method to measure the text similarity. One is Jaro-winkler distance, another is distance provided by python lib `difflib`.


```python
import pandas as pd
import numpy as np
import jieba
import difflib
import random
from cleanco import cleanco
import jaro
import string
```

#### Download Data


```python
comp=pd.read_stata(r"..\Compustat\names.dta")
shark=pd.read_excel(r"..\Factset Shark Repellent\FactSet SharkRepllent Data (Pulled 2019-11-19).xlsx").drop(index=[0,1,2,4])
shark.columns=shark.loc[3]
shark=shark.drop(index=3).reset_index().drop('index',axis=1)
bg=pd.read_stata(r"..\00_BGT_Firm_Names.dta")
#create name
c_name=pd.DataFrame(comp['conm']).astype(str)
s_name=pd.DataFrame(shark.iloc[:,7]).rename({'Company Name':'conm'}, axis=1)
bg_name=pd.DataFrame(bg.name_bgt[bg['total_postings_bgt']>50])
bg_name_full=pd.DataFrame(bg.name_bgt)
```

#### Provide cleaned versions of names
1. 
 `cleanco` processes company names, providing cleaned versions of the names by stripping away terms indicating organization type (such as "Ltd." or "Corp").  
- Using a database of organization type terms, It also provides an utility to deduce the type of organization, in terms of US/UK business entity types (ie. "limited liability company" or "non-profit"). 

- Details about this package can be found at https://pypi.org/project/cleanco/

- I also change uppercase letter to lowercase.



```python
#clean name
remove organization type and thansfer to lower case
s_name1={}.fromkeys(list(map(lambda x: cleanco(x.lower()).clean_name(), s_name.conm))).keys()
c_name1={}.fromkeys(list(map(lambda x: cleanco(x.lower()).clean_name(), c_name.conm))).keys()
bg_name1={}.fromkeys(list(map(lambda x: cleanco(x.lower()).clean_name(), bg_name.name_bgt))).keys()
```

#### Jaro-winker distance

1. Jaro-winker distance is a letter-based distance that we can use to measure the similarity between two strings
2. Here I use python package `jaro` to calculate the distance between strings, details about this package can be found at https://pypi.org/project/jaro-winkler/.
3. Time cost: 1000 times query with a dictionary contained 6000 strings will cost about 175s


```python
def jaro_distance(list1,list2):
    """
    Measure strings similarity by Jaro-winker distance.
    
    Parameters
    ----------
        list1: list of query strings
        list2: list of names dictionary       
    Returns
    -------
        df: Dataframe with three columns: "query_name", "match" and "score"
        "query_name" is the query string(target company name)
        "match" is the most similary string found at dictionary for the query string
        "score" is a float number used to measure the similarity between the query string and "match" string, range(0,1)
              
    """
    df=pd.DataFrame(list1)
    label=[]
    score_get=np.empty(len(list1))
    score=np.empty(len(list2))
    for n1,str1 in enumerate(list1):
        for n2,str2 in enumerate(list2):
            score[n2]=jaro.jaro_winkler_metric(str1,str2)
        imax=np.argmax(score)
        label.append(list2[imax])
        score_get[n1]=max(score)
    df['match']=label
    df['score']=score_get
    df.rename(columns={0:'query_name'})
    return df
```

#### StrSimilarity 

1. StrSimilarity is a function to measure string similarity on word-based and letter-based
- The alagorithm of StrSimilarity is:
 - First to count the number of common words of query string and potential matched string
 - Then keep potential strings with the n-highest common word number
 - Calculate adjusted scores for each potential matched strings(optional)
 - Used `difflib` to calculate these n strings'similarity with query string
 - The final similarity score is score given by `difflib` minutes adjustment scores
- The advantage of this method is avoiding matching query company name with those companies whose name is very similar
- Time cost: 1000 times query with a dictionary contained 6000 strings will cost about 35.3s


```python
class StrSimilarity1():
    def __init__(self,word):
        self.word=word
#Compared函数，参数str_list是对比字符串列表
#返回原始字符串分词后和对比字符串的匹配次数，返回一个字典
    def Compared(self,str_list):
        """
        Count common words.
        
        Parameters
        ----------
        str_list: a list contains potential matched strings
        
        
        Returns
        ----------
        dict_data: a dictionary
        keys are the potential matched strings
        value are the number of common words
        
        """
        dict_data={}
        sarticiple=list(self.word.strip().translate(str.maketrans('', '', string.punctuation)).split())
        for strs in str_list:
            #s_name list
            strs_word=list(strs.strip().translate(str.maketrans('', '', string.punctuation)).split())
            num=0
            for strs1 in strs_word:
                if strs1 in sarticiple:
                    num = num+1
                else: 
                    num = num
            dict_data[strs]=num
        return dict_data
    #NumChecks函数，参数dict_data是原始字符串分词后和对比字符串的匹配次数的字典，也就是Compared函数的返回值
    #返回出现次数最高的两个，返回一个字典
    def NumChecks(self,dict_data):
        """
        Return two potential strings with the hightest common word number.
        
        
        Parameters
        ----------
        dict_data: a dictionary
        keys are the potential matched strings
        value are the number of common words(return of Compared)
        
        
        Returns
        ----------
        dict_data: a dictionary
        keys are the two potential matched strings with the highest number of common words
        value are the number of common words
        
        """       
        list_data = sorted(dict_data.items(), key=lambda asd:asd[1], reverse=True)
        length = len(list_data)
        json_data = {}
        if length>=2:
            datas = list_data[:2]
        else:
            datas = list_data[:length]
        for data in datas:
            json_data[data[0]]=data[1]
        return json_data
#MMedian函数，参数dict_data是出现次数最高的两个对比字符串的字典，也就是NumChecks函数的返回值
#返回对比字符串和调节值的字典
    def MMedian(self,dict_data):
        """
        Calculate adjusted similarity scores for most potential strings(optional step).
        
        
        Parameters
        ----------
        dict_data: a dictionary
        keys are the two potential matched strings with the highest number of common words
        value are the number of common words(return of NumChecks)
        
        
        Returns
        ----------
        dict_data: a dictionary
        keys are the two potential matched strings with the highest number of common words
        value are the adjusted similarity scores
               
        """   
        
        median_list={}
        l=len(list(self.word.strip().translate(str.maketrans('', '', string.punctuation)).split()))#query string word numbers
        for k,v in dict_data.items():#k is potential string, v is the common word number
            length=len(list(k.strip().translate(str.maketrans('', '', string.punctuation)).split()))#potential string word numbers
            if l>v: 
                if v==length:
                    xx=-1
                else: 
                    xx = ((abs(l-v))/l)
            else: 
                 xx=-2    
            median_list[k] = xx
        return median_list
    
    
    
#Appear函数，参数dict_data是对比字符串和调节值的字典，也就是MMedian函数的返回值
#返回最相似的字符串
    def Appear(self,dict_data):
        """
        Return the most similar potential string.
        
        
        Parameters
        ----------
        dict_data: a dictionary
        keys are the two potential matched strings with the highest number of common words
        value are the adjusted similarity scores(return of  MMedian)
        
        
        Returns
        ----------
        dict_data: a dictionary
        key is the query string
        value is most similar potential string
               
        """   
        json_data={}
        for k,v in dict_data.items():
            fraction = difflib.SequenceMatcher(None, self.word, k).quick_ratio()-v
            json_data[k]=fraction
        tulp_data = sorted(json_data.items(), key=lambda asd:asd[1], reverse=True)
        return tulp_data[0][0],tulp_data[0][1]   
    
def name_match1(query_list1,str_list1):
    """
    Measure strings similarity by StrSimilary.
    
    Parameters
    ----------
        query_list1: list of query strings
        str_list1: list of names dictionary       
    Returns
    -------
        df: Dataframe with three columns: "query_name", "match" and "score"
        "query_name" is the query string(target company name)
        "match" is the most similary string found at dictionary for the query string
        "score" is a float number used to measure the similarity between the query string and "match" string, range(0,1)             
    """ 
    name_match=[]
    score=[]
    #str_list1=list(' '.join(str1.strip().translate(str.maketrans('', '', string.punctuation)).split()) for str1 in str_list1)
    for i,str_query in enumerate(query_list1):
        def main():
            query_str =str_query
            str_list=str_list1
    
            ss = StrSimilarity1(query_str)
            list_data = ss.Compared(str_list)
            num = ss.NumChecks(list_data)
            mmedian = ss.MMedian(num)
            #print(query_str,ss.Appear(mmedian))
            return ss.Appear(mmedian)
        if __name__=="__main__":
            name_match.append(main()[0])
            score.append(main()[1])
    df=pd.DataFrame(query_list1)
    df['match']=name_match
    df['score']=score
    df.rename(columns={0:'query_name'})
    return df
```

This is a StrSimilarity function but with different definition of "common word" and adjusted similarity scores

1. I extend the definition of "common word" (ie. 'hotels' and 'hotel' will be regarded as common word, but 'hodel' and 'hotel' will not )
- I set different penalty weights to dismatch in words and in letters


```python
# extend common word definition
def max_num(str1,str2):
    i=0
    while True:
        if str1[:len(str1)-i] in str2:
            return len(str1)-i,i
            break
        else:
            i+=1

#停用词，这里只是针对例子增加的停用词，如果数量很大可以保存在一个文件中
#stopwords=['financial','service','services','group','company','companies','the','managerment']
stopwords=[]
class StrSimilarity3():
    def __init__(self,word):
        self.word=word

#Compared函数，参数str_list是对比字符串列表
#返回原始字符串分词后和对比字符串的匹配次数，返回一个字典
    def Compared(self,str_list):
        """
        Count common words.
        
        Parameters
        ----------
        str_list: a list contains potential matched strings
        
        
        Returns
        ----------
        dict_data: a dictionary
        keys are the potential matched strings
        value are the number of common words
        
        """
        dict_data={}
        sarticiple=self.word.replace(' and ', " & ").translate(str.maketrans('', '', string.punctuation))
        for strs,strs_word in str_list.items():
            num=0
            l=0
            for strs1 in strs_word:
                lens,i=max_num(strs1,sarticiple) #uset to solve match problem 'hotel' vs. 'hotels'
                if i<=2 and lens>=3:
                    num = num+1
                else:
                    num=num
            dict_data[strs]=num
        return dict_data

    
    #NumChecks函数，参数dict_data是原始字符串分词后和对比字符串的匹配次数的字典，也就是Compared函数的返回值
    #返回出现次数最高的两个，返回一个字典
    def NumChecks(self,dict_data):
        """
        Return three potential strings with the hightest common word number.
        
        
        Parameters
        ----------
        dict_data: a dictionary
        keys are the potential matched strings
        value are the number of common words(return of Compared)
        
        
        Returns
        ----------
        dict_data: a dictionary
        keys are the three potential matched strings with the highest number of common words
        value are the number of common words
        
        """  
        list_data = sorted(dict_data.items(), key=lambda asd:asd[1], reverse=True)
        length = len(list_data)
        json_data = {}
        json_data1 = {}
        if length>=3:
            datas = list_data[:3]
        else:
            datas = list_data[:length]
        for data in datas:
            json_data[data[0]]=data[1]# match number of word
            #json_data1[data[0]]=dict_data1[data[0]]#match number of letter
        return json_data#,json_data1
    
#MMedian函数，参数dict_data是出现次数最高的两个对比字符串的字典，也就是NumChecks函数的返回值
#返回对比字符串和调节值xx的字典
       
    def MMedian(self,dict_data):
         """
        Calculate adjusted similarity scores for most potential strings(optional step).
        
        
        Parameters
        ----------
        dict_data: a dictionary
        keys are the three potential matched strings with the highest number of common words
        value are the number of common words(return of NumChecks)
        
        
        Returns
        ----------
        dict_data: a dictionary
        keys are the three potential matched strings with the highest number of common words
        value are the adjusted similarity scores
               
        """   
        median_list={}
        length = len(self.word)
        for k,v in dict_data.items():
            num = np.median([len(k),length])
            if abs(length-num) !=0 :
                xx = (abs(length - num)) * 0.017
            else:
                xx = 0
            median_list[k] = xx
        return median_list
 
    
    
#Appear函数，参数dict_data是对比字符串和调节值的字典，也就是MMedian函数的返回值
#返回最相似的字符串
    def Appear(self,dict_data):
        """
        Return the most similar potential string.
        
        
        Parameters
        ----------
        dict_data: a dictionary
        keys are the three potential matched strings with the highest number of common words
        value are the adjusted similarity scores(return of  MMedian)
        
        
        Returns
        ----------
        dict_data: a dictionary
        key is the query string
        value is most similar potential string
               
        """   
        json_data={}
        for k,v in dict_data.items():
            fraction = difflib.SequenceMatcher(None, self.word, k).quick_ratio()-v#v 调节值
            #fraction=-v
            json_data[k]=fraction
        tulp_data = sorted(json_data.items(), key=lambda asd:asd[1], reverse=True)
        return tulp_data[0][0],tulp_data[0][1]
    
def name_match3(query_list1,str_list1):
    """
    Measure strings similarity by StrSimilary.
    
    Parameters
    ----------
        query_list1: list of query strings
        str_list1: list of names dictionary       
    Returns
    -------
        df: Dataframe with three columns: "query_name", "match" and "score"
        "query_name" is the query string(target company name)
        "match" is the most similary string found at dictionary for the query string
        "score" is a float number used to measure the similarity between the query string and "match" string, range(0,1)             
    """ 
    name_match=[]
    score=[]
    #str_list1=list(' '.join(str1.strip().translate(str.maketrans('', '', string.punctuation)).split()) for str1 in str_list1)
    for i,str_query in enumerate(query_list1):
        def main():
            query_str =str_query
            str_list=str_list1
    
            ss = StrSimilarity3(query_str)
            list_data= ss.Compared(str_list)
            num= ss.NumChecks(list_data)
            mmedian = ss.MMedian(num)
            #print(query_str,ss.Appear(mmedian))
            return ss.Appear(mmedian)
        if __name__=="__main__":
            name_match.append(main()[0])
            score.append(main()[1])
    df=pd.DataFrame(query_list1)
    df['match']=name_match
    df['score']=score
    df.rename(columns={0:'query_name'})
    return df
```

 

 

### This is a match test using simulated data

1. I pick 1000 random names from `Shark Repellent` as my data label `train_Y` (because I need the true match label)
- I pick words(letters) from burning glass and add them randomly to `train_Y` to create train data `train_X`
- `train_X` now is the query list, `train_Y` is the true label od query word,  `Shark Repellent` is my potential matched string list


```python
#train
noise=random.sample(s_name1,(1000))
train_Y=sorted(random.sample(s_name1,(1000)))#true lable
train_X=list(map(lambda x,y:x+' '+y[:4]+' '+y[-3:], train_Y,noise))
```

- Below is the test data, where `real nam` is the true label, `noise name` is string waiting to match with `Shark Repellent`
- Order of words doen't matter in each algorithm


```python
data=pd.DataFrame(train_Y,columns=['real name'])
data['noise name']=train_X
data.head(-10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>real name</th>
      <th>noise name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>180 connect</td>
      <td>180 connect post ngs</td>
    </tr>
    <tr>
      <td>1</td>
      <td>22nd century group</td>
      <td>22nd century group petr ent</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3par</td>
      <td>3par tuto cal</td>
    </tr>
    <tr>
      <td>3</td>
      <td>a. m. castle</td>
      <td>a. m. castle rovi ovi</td>
    </tr>
    <tr>
      <td>4</td>
      <td>a10 networks</td>
      <td>a10 networks redk ons</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>985</td>
      <td>xenia hotels &amp; resorts</td>
      <td>xenia hotels &amp; resorts prog ial</td>
    </tr>
    <tr>
      <td>986</td>
      <td>xenoport</td>
      <td>xenoport surm ics</td>
    </tr>
    <tr>
      <td>987</td>
      <td>xo holdings</td>
      <td>xo holdings usel com</td>
    </tr>
    <tr>
      <td>988</td>
      <td>xplore technologies</td>
      <td>xplore technologies worl ngs</td>
    </tr>
    <tr>
      <td>989</td>
      <td>xzeres</td>
      <td>xzeres delp phi</td>
    </tr>
  </tbody>
</table>
<p>990 rows × 2 columns</p>
</div>



 

#### StrSimilarity1 Test Result


```python
%%time
df1_train=name_match1(train_X,str_list11)
df1_train['lable']=train_Y
df1_train.sort_values(by='score',ascending=False).head(50)
#threashold=1
df1_result=df1_train[df1_train.score>=1]
accuracy_ratio=sum(np.where(df1_result.match==df1_result.lable,1,0))/np.shape(df1_result)[0]
print(accuracy_ratio)
```

    0.9050505050505051
    Wall time: 39.3 s
    


```python
df1_train.sort_values(by='score',ascending=False).head(-10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>query_name</th>
      <th>match</th>
      <th>score</th>
      <th>lable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>18</td>
      <td>advent claymore convertible securities and inc...</td>
      <td>advent claymore convertible securities and inc...</td>
      <td>1.923077</td>
      <td>advent claymore convertible securities and inc...</td>
    </tr>
    <tr>
      <td>355</td>
      <td>federated premier intermediate municipal incom...</td>
      <td>federated premier intermediate municipal incom...</td>
      <td>1.920354</td>
      <td>federated premier intermediate municipal incom...</td>
    </tr>
    <tr>
      <td>126</td>
      <td>blackrock investment quality municipal income ...</td>
      <td>blackrock investment quality municipal income ...</td>
      <td>1.918919</td>
      <td>blackrock investment quality municipal income ...</td>
    </tr>
    <tr>
      <td>645</td>
      <td>nuveen insured florida tax-free advantage muni...</td>
      <td>nuveen insured florida taxfree advantage munic...</td>
      <td>1.916667</td>
      <td>nuveen insured florida tax-free advantage muni...</td>
    </tr>
    <tr>
      <td>552</td>
      <td>managed duration investment grade municipal fu...</td>
      <td>managed duration investment grade municipal fund</td>
      <td>1.914286</td>
      <td>managed duration investment grade municipal fund</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>649</td>
      <td>nwh smur ner</td>
      <td>nwh</td>
      <td>1.400000</td>
      <td>nwh</td>
    </tr>
    <tr>
      <td>105</td>
      <td>bab imme ion</td>
      <td>bab</td>
      <td>1.400000</td>
      <td>bab</td>
    </tr>
    <tr>
      <td>483</td>
      <td>iqe fanu nuc</td>
      <td>iqe</td>
      <td>1.400000</td>
      <td>iqe</td>
    </tr>
    <tr>
      <td>83</td>
      <td>at&amp;t sale ons</td>
      <td>att</td>
      <td>1.375000</td>
      <td>at&amp;t</td>
    </tr>
    <tr>
      <td>136</td>
      <td>bp live ile</td>
      <td>bp</td>
      <td>1.307692</td>
      <td>bp</td>
    </tr>
  </tbody>
</table>
<p>990 rows × 4 columns</p>
</div>



 

#### Jaro-winkler Distance Test Result


```python
%%time
df_jaro_train=jaro_distance(train_X,str_list11)
df_jaro_train['lable']=train_Y
df_jaro_train.sort_values(by='score',ascending=False).head(50)
#threashold
df_jaro_result=df_jaro_train[df_jaro_train.score>=0.8]
accuracy_ratio2=sum(np.where(df_jaro_result.match==df_jaro_result.lable,1,0))/np.shape(df_jaro_result)[0]
print(accuracy_ratio2)
```

    0.8986960882647944
    Wall time: 3min 29s
    


```python
df_jaro_train.sort_values(by='score',ascending=False).head(-10)
```

 

#### Choose of Threashold


```python
#choose the best threashold
def test_func1(train_X,train_Y,str_list11,threashold1):
    df1_train=name_match1(train_X,str_list11)
    df1_train['lable']=train_Y
    df1_result=df1_train[df1_train.score>=threashold1]
    accuracy_ratio=sum(np.where(df1_result.match==df1_result.lable,1,0))/np.shape(df1_result)[0]
    return accuracy_ratio

def test_func2(train_X,train_Y,str_list11,threashold2):
    df_jaro_train=jaro_distance(train_X,str_list11)
    df_jaro_train['lable']=train_Y
    df_jaro_result=df_jaro_train[df_jaro_train.score>=threashold2]
    accuracy_ratio2=sum(np.where(df_jaro_result.match==df_jaro_result.lable,1,0))/np.shape(df_jaro_result)[0]

    return accuracy_ratio2
```


```python
acc1={}
acc2={}
n=5
for threashold1,threashold2 in zip(np.linspace(0.9,1.5,10),np.linspace(0.75,0.95,10)):
    ratio1=0
    ratio2=0
    i=-n
    while i:
        noise=random.sample(s_name1,(100))
        train_Y=sorted(random.sample(s_name1,(100)))#true lable
        train_X=list(map(lambda x,y:x+' '+y[:4]+' '+y[-3:], train_Y,noise))
        ratio1=ratio1+test_func1(train_X,train_Y,str_list11,threashold1)   
        ratio2=ratio2+test_func2(train_X,train_Y,str_list11,threashold2) 
        i+=1
    acc1[threashold1]=ratio1/n
    acc2[threashold2]=ratio2/n
    
```


```python
k=acc1.keys()
v=acc1.values()
k2=acc2.keys()
v2=acc2.values()
table1=pd.DataFrame([k,v,k2,v2]).T.rename(columns={0:'threshold_strSimi',1:'accuracy_strSimi',2:'threshold_jaro',3:'accuracy_jaro'})
table1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>threshold_strSimi</th>
      <th>accuracy_strSimi</th>
      <th>threshold_jaro</th>
      <th>accuracy_jaro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.900000</td>
      <td>0.896000</td>
      <td>0.750000</td>
      <td>0.895818</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.966667</td>
      <td>0.896000</td>
      <td>0.772222</td>
      <td>0.896000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.033333</td>
      <td>0.906000</td>
      <td>0.794444</td>
      <td>0.905677</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.100000</td>
      <td>0.908000</td>
      <td>0.816667</td>
      <td>0.909448</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.166667</td>
      <td>0.904000</td>
      <td>0.838889</td>
      <td>0.903196</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.233333</td>
      <td>0.912000</td>
      <td>0.861111</td>
      <td>0.927856</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.300000</td>
      <td>0.910000</td>
      <td>0.883333</td>
      <td>0.915960</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.366667</td>
      <td>0.902000</td>
      <td>0.905556</td>
      <td>0.932159</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.433333</td>
      <td>0.908775</td>
      <td>0.927778</td>
      <td>0.933825</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.500000</td>
      <td>0.905708</td>
      <td>0.950000</td>
      <td>0.933782</td>
    </tr>
  </tbody>
</table>
</div>



 

#### Test Result for Real Data (use 1000 sample)


```python
#real data test
random.seed(123)
query_list1=random.sample(bg_name1,1000)#1000 names 
s_name1=sorted(s_name1, key=len)    
str_list11=list(' '.join(str1.strip().translate(str.maketrans('', '', string.punctuation)).split()) for str1 in s_name1)
query_list1=list(' '.join(str1.strip().translate(str.maketrans('', '', string.punctuation)).split()) for str1 in query_list1)
query_list1.sort(reverse=False)
```

 

#### StrSimilarity1 Test Result


```python
%%time
df1=name_match1(query_list1,str_list11).sort_values(by='score',ascending=False)
```

    Wall time: 37 s
    


```python
#threshold=1
df1[df1.score>1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>941</td>
      <td>universal technical institute</td>
      <td>universal technical institute</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>512</td>
      <td>kensey nash</td>
      <td>kensey nash</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>630</td>
      <td>national interstate</td>
      <td>national interstate</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>813</td>
      <td>smc corp america</td>
      <td>smc</td>
      <td>1.315789</td>
    </tr>
    <tr>
      <td>143</td>
      <td>boca west country club</td>
      <td>west</td>
      <td>1.307692</td>
    </tr>
    <tr>
      <td>111</td>
      <td>ball factory indoor play cafe</td>
      <td>ball</td>
      <td>1.242424</td>
    </tr>
    <tr>
      <td>634</td>
      <td>netvision resources nvr</td>
      <td>nvr</td>
      <td>1.230769</td>
    </tr>
    <tr>
      <td>366</td>
      <td>franklin west supervisory union</td>
      <td>west</td>
      <td>1.228571</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 3 columns</p>
</div>




```python
#threshold=1.5
df1[df1.score>1.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>941</td>
      <td>universal technical institute</td>
      <td>universal technical institute</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>512</td>
      <td>kensey nash</td>
      <td>kensey nash</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>630</td>
      <td>national interstate</td>
      <td>national interstate</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>856</td>
      <td>superdry</td>
      <td>superdry</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>139</td>
      <td>bmc software</td>
      <td>bmc software</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>204</td>
      <td>christopher banks</td>
      <td>christopher banks</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>90</td>
      <td>ashland</td>
      <td>ashland</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>196</td>
      <td>charter communications</td>
      <td>charter communications</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>689</td>
      <td>packeteer</td>
      <td>packeteer</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>978</td>
      <td>westmoreland</td>
      <td>westmoreland coal</td>
      <td>2.827586</td>
    </tr>
    <tr>
      <td>610</td>
      <td>millennium group</td>
      <td>millennium services group</td>
      <td>2.780488</td>
    </tr>
    <tr>
      <td>495</td>
      <td>it solutions</td>
      <td>pomeroy it solutions</td>
      <td>2.750000</td>
    </tr>
    <tr>
      <td>896</td>
      <td>timken</td>
      <td>the timken</td>
      <td>2.750000</td>
    </tr>
    <tr>
      <td>577</td>
      <td>management services</td>
      <td>birner dental management services</td>
      <td>2.730769</td>
    </tr>
    <tr>
      <td>387</td>
      <td>georgetown</td>
      <td>georgetown bancorp</td>
      <td>2.714286</td>
    </tr>
    <tr>
      <td>303</td>
      <td>easylink services</td>
      <td>easylink services international</td>
      <td>2.708333</td>
    </tr>
    <tr>
      <td>861</td>
      <td>sybron dental</td>
      <td>sybron dental specialties</td>
      <td>2.684211</td>
    </tr>
    <tr>
      <td>161</td>
      <td>brooke</td>
      <td>brooke group</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <td>727</td>
      <td>pulse</td>
      <td>pulse data</td>
      <td>2.666667</td>
    </tr>
    <tr>
      <td>508</td>
      <td>juniper</td>
      <td>juniper networks</td>
      <td>2.608696</td>
    </tr>
    <tr>
      <td>425</td>
      <td>harvard</td>
      <td>harvard bioscience</td>
      <td>2.560000</td>
    </tr>
    <tr>
      <td>993</td>
      <td>worldgate</td>
      <td>worldgate communications</td>
      <td>2.545455</td>
    </tr>
    <tr>
      <td>229</td>
      <td>cohen steers</td>
      <td>cohen steers select utility fund</td>
      <td>2.545455</td>
    </tr>
    <tr>
      <td>325</td>
      <td>exact</td>
      <td>exact sciences</td>
      <td>2.526316</td>
    </tr>
    <tr>
      <td>745</td>
      <td>reynolds packaging</td>
      <td>the reynolds reynolds</td>
      <td>2.512821</td>
    </tr>
    <tr>
      <td>913</td>
      <td>triad</td>
      <td>triad hospitals</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <td>881</td>
      <td>tenet</td>
      <td>tenet healthcare</td>
      <td>2.476190</td>
    </tr>
    <tr>
      <td>136</td>
      <td>blue sky</td>
      <td>blue sky alternative investments</td>
      <td>2.400000</td>
    </tr>
    <tr>
      <td>511</td>
      <td>kennedy</td>
      <td>kennedy wilson europe real estate</td>
      <td>2.350000</td>
    </tr>
    <tr>
      <td>638</td>
      <td>new technology solutions</td>
      <td>technology solutions</td>
      <td>1.909091</td>
    </tr>
    <tr>
      <td>685</td>
      <td>pacific american fish</td>
      <td>american pacific</td>
      <td>1.864865</td>
    </tr>
    <tr>
      <td>875</td>
      <td>teamstaff gs</td>
      <td>teamstaff</td>
      <td>1.857143</td>
    </tr>
    <tr>
      <td>241</td>
      <td>conperio technology solutions</td>
      <td>technology solutions</td>
      <td>1.816327</td>
    </tr>
    <tr>
      <td>522</td>
      <td>kitty hawk kites</td>
      <td>kitty hawk</td>
      <td>1.769231</td>
    </tr>
    <tr>
      <td>234</td>
      <td>comfort systems usa midatlantic</td>
      <td>comfort systems usa</td>
      <td>1.760000</td>
    </tr>
    <tr>
      <td>296</td>
      <td>e mds</td>
      <td>mds</td>
      <td>1.750000</td>
    </tr>
    <tr>
      <td>940</td>
      <td>universal parking</td>
      <td>universal</td>
      <td>1.692308</td>
    </tr>
    <tr>
      <td>206</td>
      <td>citizens for citizens</td>
      <td>citizens</td>
      <td>1.551724</td>
    </tr>
    <tr>
      <td>809</td>
      <td>sky zone</td>
      <td>sky</td>
      <td>1.545455</td>
    </tr>
    <tr>
      <td>337</td>
      <td>farrow ball</td>
      <td>ball</td>
      <td>1.533333</td>
    </tr>
    <tr>
      <td>397</td>
      <td>gogo squeez</td>
      <td>gogo</td>
      <td>1.533333</td>
    </tr>
    <tr>
      <td>874</td>
      <td>team sewell</td>
      <td>team</td>
      <td>1.533333</td>
    </tr>
    <tr>
      <td>128</td>
      <td>berkshire hathaway homeservices homesale realt...</td>
      <td>berkshire hathaway</td>
      <td>1.529412</td>
    </tr>
    <tr>
      <td>939</td>
      <td>universal atlantic systems</td>
      <td>universal</td>
      <td>1.514286</td>
    </tr>
  </tbody>
</table>
</div>



#### StrSimilarity2 Test Result


```python
%%time
df2=name_match3(query_list1,str_list1).sort_values(by='score',ascending=False)
```

    Wall time: 1min
    


```python
#threshold=0.8
df2[df2.score>0.8]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>196</td>
      <td>charter communications</td>
      <td>charter communications</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>630</td>
      <td>national interstate</td>
      <td>national interstate</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>856</td>
      <td>superdry</td>
      <td>superdry</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>512</td>
      <td>kensey nash</td>
      <td>kensey nash</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>139</td>
      <td>bmc software</td>
      <td>bmc software</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>941</td>
      <td>universal technical institute</td>
      <td>universal technical institute</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>90</td>
      <td>ashland</td>
      <td>ashland</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>689</td>
      <td>packeteer</td>
      <td>packeteer</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>204</td>
      <td>christopher banks</td>
      <td>christopher &amp; banks</td>
      <td>0.927444</td>
    </tr>
    <tr>
      <td>595</td>
      <td>mdu resources</td>
      <td>mod resources</td>
      <td>0.923077</td>
    </tr>
    <tr>
      <td>776</td>
      <td>sanimax</td>
      <td>animas</td>
      <td>0.914577</td>
    </tr>
    <tr>
      <td>830</td>
      <td>spn well services</td>
      <td>us well services</td>
      <td>0.900591</td>
    </tr>
    <tr>
      <td>641</td>
      <td>ngmoco</td>
      <td>mocon</td>
      <td>0.900591</td>
    </tr>
    <tr>
      <td>65</td>
      <td>amica</td>
      <td>amicas</td>
      <td>0.900591</td>
    </tr>
    <tr>
      <td>395</td>
      <td>gm financial group</td>
      <td>si financial group</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <td>638</td>
      <td>new technology solutions</td>
      <td>technology solutions</td>
      <td>0.875091</td>
    </tr>
    <tr>
      <td>338</td>
      <td>fbl financial</td>
      <td>tf financial</td>
      <td>0.871500</td>
    </tr>
    <tr>
      <td>850</td>
      <td>sunbelt management associates</td>
      <td>health management associates</td>
      <td>0.868693</td>
    </tr>
    <tr>
      <td>948</td>
      <td>va information technologies</td>
      <td>china information technology</td>
      <td>0.864227</td>
    </tr>
    <tr>
      <td>603</td>
      <td>memorial health care</td>
      <td>mariner health care</td>
      <td>0.863295</td>
    </tr>
    <tr>
      <td>915</td>
      <td>triple crown services</td>
      <td>trico marine services</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <td>865</td>
      <td>systems technologies</td>
      <td>ems technologies</td>
      <td>0.854889</td>
    </tr>
    <tr>
      <td>471</td>
      <td>ifb solutions</td>
      <td>pdf solutions</td>
      <td>0.846154</td>
    </tr>
    <tr>
      <td>241</td>
      <td>conperio technology solutions</td>
      <td>cognizant technology solutions</td>
      <td>0.838958</td>
    </tr>
    <tr>
      <td>884</td>
      <td>teton services and management</td>
      <td>birner dental management services</td>
      <td>0.836968</td>
    </tr>
    <tr>
      <td>391</td>
      <td>global commerce services</td>
      <td>premiere global services</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <td>593</td>
      <td>mb global logistics</td>
      <td>echo global logistics</td>
      <td>0.833000</td>
    </tr>
    <tr>
      <td>875</td>
      <td>teamstaff gs</td>
      <td>teamstaff</td>
      <td>0.831643</td>
    </tr>
    <tr>
      <td>933</td>
      <td>united global solutions</td>
      <td>moduslink global solutions</td>
      <td>0.831643</td>
    </tr>
    <tr>
      <td>862</td>
      <td>synergroup systems</td>
      <td>sierra systems group</td>
      <td>0.825105</td>
    </tr>
    <tr>
      <td>329</td>
      <td>exilant technologies</td>
      <td>exide technologies</td>
      <td>0.825105</td>
    </tr>
    <tr>
      <td>295</td>
      <td>e computer technologies</td>
      <td>compex technologies</td>
      <td>0.823143</td>
    </tr>
    <tr>
      <td>685</td>
      <td>pacific american fish</td>
      <td>american pacific</td>
      <td>0.822365</td>
    </tr>
    <tr>
      <td>13</td>
      <td>abps healthcare</td>
      <td>hca healthcare</td>
      <td>0.819086</td>
    </tr>
    <tr>
      <td>41</td>
      <td>alans group</td>
      <td>alamo group</td>
      <td>0.818182</td>
    </tr>
    <tr>
      <td>126</td>
      <td>benjamin west</td>
      <td>west marine</td>
      <td>0.816333</td>
    </tr>
    <tr>
      <td>615</td>
      <td>mohegan holding</td>
      <td>gam holding</td>
      <td>0.812154</td>
    </tr>
    <tr>
      <td>97</td>
      <td>aureus medical group</td>
      <td>journal media group</td>
      <td>0.812013</td>
    </tr>
    <tr>
      <td>489</td>
      <td>international business college</td>
      <td>international business machines</td>
      <td>0.811172</td>
    </tr>
    <tr>
      <td>542</td>
      <td>lee company tn</td>
      <td>telia company</td>
      <td>0.806315</td>
    </tr>
    <tr>
      <td>942</td>
      <td>universal truckload services</td>
      <td>universal health services</td>
      <td>0.804689</td>
    </tr>
    <tr>
      <td>64</td>
      <td>amerihealth caritas</td>
      <td>sabra health care reit</td>
      <td>0.803768</td>
    </tr>
    <tr>
      <td>706</td>
      <td>plymouth rock group of companies</td>
      <td>the interpublic group of companies</td>
      <td>0.801182</td>
    </tr>
    <tr>
      <td>686</td>
      <td>pacific data integrators</td>
      <td>asia pacific data centre group</td>
      <td>0.800852</td>
    </tr>
    <tr>
      <td>347</td>
      <td>first community mortgage</td>
      <td>community first bancorp</td>
      <td>0.800011</td>
    </tr>
  </tbody>
</table>
</div>




```python
#threshold=0.9
df2[df2.score>0.85]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>196</td>
      <td>charter communications</td>
      <td>charter communications</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>630</td>
      <td>national interstate</td>
      <td>national interstate</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>856</td>
      <td>superdry</td>
      <td>superdry</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>512</td>
      <td>kensey nash</td>
      <td>kensey nash</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>139</td>
      <td>bmc software</td>
      <td>bmc software</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>941</td>
      <td>universal technical institute</td>
      <td>universal technical institute</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>90</td>
      <td>ashland</td>
      <td>ashland</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>689</td>
      <td>packeteer</td>
      <td>packeteer</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>204</td>
      <td>christopher banks</td>
      <td>christopher &amp; banks</td>
      <td>0.927444</td>
    </tr>
    <tr>
      <td>595</td>
      <td>mdu resources</td>
      <td>mod resources</td>
      <td>0.923077</td>
    </tr>
    <tr>
      <td>776</td>
      <td>sanimax</td>
      <td>animas</td>
      <td>0.914577</td>
    </tr>
    <tr>
      <td>830</td>
      <td>spn well services</td>
      <td>us well services</td>
      <td>0.900591</td>
    </tr>
    <tr>
      <td>641</td>
      <td>ngmoco</td>
      <td>mocon</td>
      <td>0.900591</td>
    </tr>
    <tr>
      <td>65</td>
      <td>amica</td>
      <td>amicas</td>
      <td>0.900591</td>
    </tr>
    <tr>
      <td>395</td>
      <td>gm financial group</td>
      <td>si financial group</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <td>638</td>
      <td>new technology solutions</td>
      <td>technology solutions</td>
      <td>0.875091</td>
    </tr>
    <tr>
      <td>338</td>
      <td>fbl financial</td>
      <td>tf financial</td>
      <td>0.871500</td>
    </tr>
    <tr>
      <td>850</td>
      <td>sunbelt management associates</td>
      <td>health management associates</td>
      <td>0.868693</td>
    </tr>
    <tr>
      <td>948</td>
      <td>va information technologies</td>
      <td>china information technology</td>
      <td>0.864227</td>
    </tr>
    <tr>
      <td>603</td>
      <td>memorial health care</td>
      <td>mariner health care</td>
      <td>0.863295</td>
    </tr>
    <tr>
      <td>915</td>
      <td>triple crown services</td>
      <td>trico marine services</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <td>865</td>
      <td>systems technologies</td>
      <td>ems technologies</td>
      <td>0.854889</td>
    </tr>
  </tbody>
</table>
</div>



 

#### Jaro-winkler Test Result


```python
%%time
df_jaro=jaro_distance(query_list1,str_list11).sort_values(by='score',ascending=False)
```

    Wall time: 2min 55s
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>856</td>
      <td>superdry</td>
      <td>superdry</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>90</td>
      <td>ashland</td>
      <td>ashland</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>689</td>
      <td>packeteer</td>
      <td>packeteer</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>584</td>
      <td>marquardt transportation</td>
      <td>apartment trust of america</td>
      <td>0.721510</td>
    </tr>
    <tr>
      <td>967</td>
      <td>waldorf astoria park city</td>
      <td>selas corporation of america</td>
      <td>0.721429</td>
    </tr>
    <tr>
      <td>501</td>
      <td>jenison public schools</td>
      <td>pulse biosciences</td>
      <td>0.721390</td>
    </tr>
    <tr>
      <td>72</td>
      <td>apackansas</td>
      <td>japan asset marketing co</td>
      <td>0.721296</td>
    </tr>
    <tr>
      <td>628</td>
      <td>nacho daddy</td>
      <td>alcoa</td>
      <td>0.721212</td>
    </tr>
  </tbody>
</table>
<p>950 rows × 3 columns</p>
</div>




```python
#threshold=0.8
df_jaro[df_jaro.score>0.9]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>856</td>
      <td>superdry</td>
      <td>superdry</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>90</td>
      <td>ashland</td>
      <td>ashland</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>689</td>
      <td>packeteer</td>
      <td>packeteer</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>861</td>
      <td>sybron dental</td>
      <td>sybron dental specialties</td>
      <td>0.904000</td>
    </tr>
    <tr>
      <td>675</td>
      <td>orlando group</td>
      <td>ocado group</td>
      <td>0.903497</td>
    </tr>
    <tr>
      <td>544</td>
      <td>legacy ventures</td>
      <td>legacy reserves</td>
      <td>0.903333</td>
    </tr>
    <tr>
      <td>402</td>
      <td>great american restaurants</td>
      <td>great american group</td>
      <td>0.900769</td>
    </tr>
    <tr>
      <td>500</td>
      <td>jefferson parish</td>
      <td>jefferson bancshares</td>
      <td>0.900714</td>
    </tr>
  </tbody>
</table>
<p>73 rows × 3 columns</p>
</div>




```python
#threshold=0.8
df_jaro[df_jaro.score>0.93]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>match</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>381</td>
      <td>gateway</td>
      <td>gateway</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>532</td>
      <td>lake shore bancorp</td>
      <td>lake shore bancorp</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>856</td>
      <td>superdry</td>
      <td>superdry</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>90</td>
      <td>ashland</td>
      <td>ashland</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>689</td>
      <td>packeteer</td>
      <td>packeteer</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>139</td>
      <td>bmc software</td>
      <td>bmc software</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>204</td>
      <td>christopher banks</td>
      <td>christopher banks</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>196</td>
      <td>charter communications</td>
      <td>charter communications</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>512</td>
      <td>kensey nash</td>
      <td>kensey nash</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>941</td>
      <td>universal technical institute</td>
      <td>universal technical institute</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>630</td>
      <td>national interstate</td>
      <td>national interstate</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>65</td>
      <td>amica</td>
      <td>amicas</td>
      <td>0.966667</td>
    </tr>
    <tr>
      <td>595</td>
      <td>mdu resources</td>
      <td>mod resources</td>
      <td>0.953846</td>
    </tr>
    <tr>
      <td>875</td>
      <td>teamstaff gs</td>
      <td>teamstaff</td>
      <td>0.950000</td>
    </tr>
    <tr>
      <td>646</td>
      <td>noel group</td>
      <td>noble group</td>
      <td>0.949091</td>
    </tr>
    <tr>
      <td>523</td>
      <td>kl industries</td>
      <td>sl industries</td>
      <td>0.948718</td>
    </tr>
    <tr>
      <td>449</td>
      <td>honey wess international</td>
      <td>honeywell international</td>
      <td>0.948085</td>
    </tr>
    <tr>
      <td>862</td>
      <td>synergroup systems</td>
      <td>synergx systems</td>
      <td>0.942222</td>
    </tr>
    <tr>
      <td>978</td>
      <td>westmoreland</td>
      <td>westmoreland coal</td>
      <td>0.941176</td>
    </tr>
    <tr>
      <td>830</td>
      <td>spn well services</td>
      <td>us well services</td>
      <td>0.939951</td>
    </tr>
    <tr>
      <td>78</td>
      <td>apr energy</td>
      <td>pyr energy</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <td>490</td>
      <td>international cars</td>
      <td>international coal group</td>
      <td>0.930556</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
