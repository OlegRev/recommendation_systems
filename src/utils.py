import pandas as pd
import numpy as np
from src.recommenders import MainRecommender

def prefilter_items(data, take_n_popular=5000, users_features=None, item_features=None):
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999


    return data


def preparing_data(data_train_ranker, item_features, user_features, USER_COL='user_id', ITEM_COL='item_id', N_PREDICT=50):
    # взяли пользователей из трейна для ранжирования
    df_match_candidates = pd.DataFrame(data_train_ranker[USER_COL].unique())
    df_match_candidates.columns = [USER_COL]
    # собираем кандитатов с первого этапа (matcher)
    df_match_candidates['candidates'] = df_match_candidates[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
    
    df_items = df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    df_items.name = ITEM_COL
    df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)
    
    df_ranker_train = data_train_ranker[[USER_COL, ITEM_COL]].copy()
    df_ranker_train['target'] = 1 
    
    df_ranker_train = df_match_candidates.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')
    
    df_ranker_train['target'].fillna(0, inplace= True)
    
    df_ranker_train = df_ranker_train.merge(item_features, on='item_id', how='left')
    df_ranker_train = df_ranker_train.merge(user_features, on='user_id', how='left')
    
    X_train = df_ranker_train.drop('target', axis=1)
    y_train = df_ranker_train[['target']]
    
    return X_train, y_train


def user_feature_engineering(data, user_features, USER_COL='user_id', ITEM_COL='item_id'):
    new_user_features = user_features.merge(data, on=USER_COL, how='left')
    
    # сумма объема покупок относительно юзера
    user_features = user_features.merge(new_user_features.groupby(by=USER_COL).agg('sales_value').sum().rename('sum_user_sales_value'), on=USER_COL, how='left')
        
    # количество транзакций у пользователя
    user_features = user_features.merge(new_user_features.groupby([USER_COL])['trans_time'].count().rename('count_transactions_user'), on=USER_COL, how='left')
    
    # средний чек на одну транзакцию
    user_features = user_features.merge((new_user_features.groupby([USER_COL])["sales_value"].sum()/new_user_features.groupby([USER_COL])["trans_time"].count()).rename('average_check_per_transaction'), on=USER_COL, how='left')
    
    # количество транзакция в неделю
    user_features = user_features.merge((new_user_features.groupby([USER_COL])['trans_time'].count()/new_user_features.week_no.nunique()).rename('count_transactions_per_week'), on=USER_COL, how='left')
    
    # количество уникальных товаров купленных юзером
    user_features = user_features.merge(new_user_features.groupby(by=USER_COL).agg(ITEM_COL).nunique().rename('nuniq_item'), on=USER_COL, how='left')

    # mean, min, max, std, median кол-ва уникальных товаров в транзакции юзера
    df = new_user_features.groupby(by=[USER_COL, 'trans_time']).agg(ITEM_COL).nunique().reset_index()
    
    user_features = user_features.merge(df.groupby(by=USER_COL).agg(ITEM_COL).mean().rename('mean_n_unique_items_transaction'), on=USER_COL, how='left')

    user_features = user_features.merge(df.groupby(by=USER_COL).agg(ITEM_COL).min().rename('min_n_unique_items_transaction'), on=USER_COL, how='left')
    
    user_features = user_features.merge(df.groupby(by=USER_COL).agg(ITEM_COL).max().rename('max_n_unique_items_transaction'), on=USER_COL, how='left')
    
    user_features = user_features.merge(df.groupby(by=USER_COL).agg(ITEM_COL).std().rename('std_n_unique_items_transaction'), on=USER_COL, how='left')
    
    user_features = user_features.merge(df.groupby(by=USER_COL).agg(ITEM_COL).median().rename('median_n_unique_items_transaction'), on=USER_COL, how='left')
    
     # эмбеддинги товаров
    recommender = MainRecommender(new_user_features)
    df = recommender.model.item_factors
    n_factors = recommender.model.factors
    ind = list(recommender.id_to_itemid.values())
    df = pd.DataFrame(df, index=ind).reset_index()
    df.columns = ['item_id'] + ['factor_' + str(i + 1) for i in range(n_factors)]
    user_item_features = user_item_features.merge(df, on=['item_id'])
    
    # эмбеддинги пользователей
    df = recommender.model.user_factors
    ind = list(recommender.id_to_userid.values())
    df = pd.DataFrame(df, index=ind).reset_index()
    df.columns = ['user_id'] + ['user_factor_' + str(i + 1) for i in range(n_factors)]
    user_item_features = user_item_features.merge(df, on=['user_id'])
    
    return user_features
    
    
def item_feature_engineering(data, item_features, ITEM_COL='item_id'):
    
    new_item_features = item_features.merge(data, on=ITEM_COL, how='left')
    
    # объем продаж относительно товара
    item_features = item_features.merge(new_item_features.groupby(by=ITEM_COL).agg('sales_value').sum().rename('item_sum_sales_value'), on=ITEM_COL, how='left')
    
    # cумма количества относительно товара
    item_features = item_features.merge(new_item_features.groupby(by=ITEM_COL).agg('quantity').sum().rename('item_sum_quantity'), on=ITEM_COL, how='left')
    
    # средний объем продаж товаров в неделю
    item_features = item_features.merge((new_item_features.groupby(by=ITEM_COL).agg('sales_value').sum()/data.week_no.nunique()).rename('avg_sales_value_per_week'), on=ITEM_COL, how='left')
    
    # цена товара
    item_features = item_features.merge((new_item_features.groupby(by=ITEM_COL).agg('sales_value').sum()/new_item_features.groupby(by=ITEM_COL).agg('quantity').sum()).rename('item_price'), on=ITEM_COL, how='left')

    # Количество юзеров покупавших данный товар
    item_features = item_features.merge(new_item_features.groupby(by=ITEM_COL).agg(USER_COL).count().rename('count_users_bought_item'), on=ITEM_COL, how='left')

    return item_features


def get_replace_dict(re_list):
    replace_dict = {}
    numerator = 0
    denominator = len([itm for itm in re_list if '-' in itm])
    denominator = denominator if denominator > 0 else 1
    for itm in re_list:
        strip_itm = itm.strip('+')
        strip_itm = strip_itm.strip('K')
        try:
            replace_dict[itm] = round(int(strip_itm) + (numerator/denominator))
        except:
            if '-' in strip_itm:
                _itm = [int(_) for _ in strip_itm.split('-')]
                numerator  += abs(_itm[1]-_itm[0])
                replace_dict[itm] = round(_itm[1] - abs(_itm[1]-_itm[0])/2)
            elif '/' in itm:
                replace_dict[itm] = 0
            else:            
                replace_dict[itm] = round(int(strip_itm.split(' ')[1]) - (numerator/denominator)/5)
    return replace_dict


def get_candidates():
    pass


def postfilter_items(user_id, recommednations):
    pass