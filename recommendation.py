# -*- coding: utf-8 -*-
def preprocess():
    import pandas as pd
    import math
    import numpy as np 
            
    data_users = pd.read_csv('users_tag.csv',index_col=0)
    data_business = pd.read_csv('business_Nora.csv',index_col=0)
    data_review = pd.read_csv('reviews_cleaned.csv',index_col = 0)        
            
    data_users.review_count = pd.Series([math.log(x+1) for x in data_users.review_count])
    data_users.useful =  pd.Series([math.log(x+1) for x in data_users.useful])  
            
    #cleam business skewness
    data_business.review_count =  pd.Series([math.log(x+1) for x in data_business.review_count])        
            
    from lightfm.data import Dataset        
            
    #model establishment
    dataset = Dataset()
    dataset.fit(data_review.user_id,data_review.business_id)
    type(dataset)
    num_users, num_items = dataset.interactions_shape()        
            
    # fit item and user features. 
    dataset.fit_partial(items=data_business.business_id,
                        item_features=['stars'])
            
            
    dataset.fit_partial(items=data_business.business_id,
                        item_features=['review_count'])        
            
    tar_cols = [x for x in data_business.columns[24:]] 
            
    dataset.fit_partial(items = data_business.business_id,
                       item_features = tar_cols)        
            
    user_cols = [x for x in data_users[['review_count', 'useful',
                                       'Ice Cream & Frozen Yogurt', 'Korean', 'Tapas/Small Plates',
           'Vietnamese', 'Vegan', 'Caribbean', 'Food Delivery Services', 'Lounges',
           'Pubs', 'Greek', 'Cocktail Bars', 'Mexican', 'Wine Bars', 'Tea Rooms',
           'Delis', 'Vegetarian', 'Ethnic Food', 'Salad', 'Seafood', 'Beer',
           'American (New)', 'Juice Bars & Smoothies', 'Shopping', 'Barbeque',
           'Sports Bars', 'French', 'Chicken Wings', 'Gastropubs', 'Diners',
           'Gluten-Free', 'Thai', 'Comfort Food', 'Health Markets', 'Halal',
           'Caterers', 'Arts & Entertainment']]]        
            
    dataset.fit_partial(users=data_users.user_id,
                        user_features = user_cols)  
          
    print("Building Interactions")        
    (interactions, weights) = dataset.build_interactions([(x['user_id'],
                                                           x['business_id'],
                                                           x['stars']) for index,x in data_review.iterrows()])   
    print("Interactions Build")        
    # build user and item features
    
    def build_dict(df,tar_cols,val_list):
        rst = {}
        for col in tar_cols:
            rst[col] = df[col]
        sum_val = sum(list(rst.values())) # get sum of all the tfidf values
        
        if(sum_val == 0):
            return rst
        else:
            
            w = (2-sum(val_list))/sum_val # weight for each tag to be able to sum to 1
            for key,value in rst.items():
                rst[key] = value * w
        return rst
    
    def user_build_dict(df,tar_cols,val_list):
        rst = {}
        for col in tar_cols:
            rst[col] = df[col]
        sum_val = sum(list(rst.values())) # get sum of all the tfidf values
        
        if(sum_val == 0):
            return rst
        else:
            w = (2-sum(val_list))/sum_val # weight for each tag to be able to sum to 1
            for key,value in rst.items():
                rst[key] = value * w
        return rst
    
    # get max of each column to regularize value to [0,1]
    max_star = max(data_business.stars)
    max_b_rc = max(data_business.review_count)
    print('max_b_rc')
    print(max_b_rc)
    
    # give CF info weight 0.5, all other 0.5. Then in others, give (star, review count) 0.25 and tags 0.25
    item_features = dataset.build_item_features(((x['business_id'], 
                                                  {'stars':0.5*x['stars']/max_star,
                                                   'review_count':0.5*x['review_count']/max_b_rc,
                                                   **build_dict(x,tar_cols,[0.5*x['stars']/max_star,
                                                               0.5*x['review_count']/max_b_rc])})
                                                  for index,x in data_business.iterrows()))
    
    
    # user_features = dataset.build_user_features(((x['user_id'],
    #                                              [x['is_elite'],x['year']])
    #                                            for index, x in data_users.iterrows()))
    max_u_rc = max(data_users.review_count)
    max_useful = max(data_users.useful)
    user_features = dataset.build_user_features(((x['user_id'],
                                                 {'review_count':0.35*x['review_count']/max_u_rc,
                                                  'useful':0.35*x['useful']/max_useful,
                                                 **user_build_dict(x,user_cols,[0.35*x['review_count']/max_u_rc,0.35*x['useful']/max_useful])}) for index, x in data_users.iterrows()))
            
    #train-test split
    
    # seed = 12345 #has multiple seeds set up to account for split biases
    # seed = 101
    # seed = 186
    seed = 123
    from lightfm.cross_validation import random_train_test_split
    train,test=random_train_test_split(interactions,test_percentage=0.2,random_state=np.random.RandomState(seed))
    
    print('The dataset has %s users and %s items, '
          'with %s interactions in the test and %s interactions in the training set.'
          % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))
    
    train.multiply(test).nnz == 0 # make sure train and test are truly disjoint        
    return train,test,data_business,dataset,user_features,item_features   

list1 = preprocess()    
import pickle        
with open('rr_model.pkl', 'rb') as f:
    model1 = pickle.load(f)
        
import Rec_fx as rf        
        
#test corresponding recpmmendation

a,b = rf.sample_train_recommendation(model1,list1[0],list1[2],[74],5,'name',mapping=list1[3].mapping()[2],tag='category',
                              user_features = list1[4],item_features=list1[5])
user_index=list(set(rf.get_user_index(test)))
rf.sample_test_recommendation(model1,train,test,data_business,[user_index[51]],5,'name',mapping=dataset.mapping()[2],
                              train_interactions=train,tag='category',user_features = user_features,item_features=item_features)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        




