import pandas as pd
import numpy as np
from src.recommenders import MainRecommender

USER_COL='user_id'
ITEM_COL='item_id'

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


def set_user_item_features(data,  user_features, item_features):
    # час транзакции
    df = data.copy()
    df['hour_t'] = df.trans_time // 100
    
    # медиана часа транзакции
    user_item_features = df.groupby(by=[USER_COL, ITEM_COL]).agg('hour_t').median().rename('median_hour_transactions').reset_index()
    
    # медиана дня недели транзакции
    df['weekday_t']= df.day % 7
    _df = df.groupby(by=[USER_COL, ITEM_COL]).agg('weekday_t').median().rename('median_weekday_transactions').reset_index()
    user_item_features = user_item_features.merge(_df, on=[USER_COL, ITEM_COL], how='left')
    
    # сумма объема покупок относительно юзера
    _df = df.groupby(by=USER_COL).agg('sales_value').sum().rename('sum_user_sales_value').reset_index()
    user_item_features = user_item_features.merge(_df, on=[USER_COL], how='left')

    # кол-во товаров клиента
    _df = df.groupby([USER_COL])[ITEM_COL].count().rename('item_count').reset_index()
    user_item_features = user_item_features.merge(_df, on=[USER_COL])
    
    # количество транзакций у пользователя
    _df = df.groupby([USER_COL])['trans_time'].count().rename('count_transactions_user').reset_index()
    user_item_features = user_item_features.merge(_df, on=[USER_COL], how='left')
    
    # средний чек на одну транзакцию
    _df = df.groupby([USER_COL, 'basket_id'])['sales_value'].sum().reset_index()
    _df = _df.groupby(USER_COL)['sales_value'].mean().rename('mean_check_in_basket').reset_index()
    user_item_features = user_item_features.merge(_df, on=[USER_COL], how='left')
    
    # количество транзакция в неделю
    # _df = (df.groupby([USER_COL])['trans_time'].count()/df.week_no.nunique()).rename('count_transactions_per_week').reset_index()
    # user_item_features = user_item_features.merge(_df, on=[USER_COL], how='left')
    
    # количество уникальных товаров купленных юзером
    _df = df.groupby(by=USER_COL).agg(ITEM_COL).nunique().rename('nuniq_item').reset_index()
    user_item_features = user_item_features.merge(_df, on=[USER_COL], how='left')
    
    # количество магазинов в которых продавался товар
    _df = df.groupby([ITEM_COL])['store_id'].nunique().rename('nuniq_stores').reset_index()
    user_item_features = user_item_features.merge(_df, on=[ITEM_COL], how='left')
    
    # кол-ва уникальных товаров в корзине юзера
    _df = df.groupby([USER_COL, 'basket_id'])[ITEM_COL].nunique().reset_index()
    # mean
    __df = _df.groupby(by=USER_COL).agg(ITEM_COL).mean().rename('mean_nunique_items_in_basket').reset_index()
    user_item_features = user_item_features.merge(__df, on=[USER_COL], how='left')
    # min
    # __df = _df.groupby(by=USER_COL).agg(ITEM_COL).min().rename('min_nunique_items_in_basket').reset_index()
    # user_item_features = user_item_features.merge(__df, on=[USER_COL], how='left')
    # max 
    __df = _df.groupby(by=USER_COL).agg(ITEM_COL).max().rename('max_nunique_items_in_basket').reset_index()
    user_item_features = user_item_features.merge(__df, on=[USER_COL], how='left')
    # std
    __df = _df.groupby(by=USER_COL).agg(ITEM_COL).std().rename('std_nunique_items_in_basket').reset_index()
    user_item_features = user_item_features.merge(__df, on=[USER_COL], how='left')
    # median
    # __df = _df.groupby(by=USER_COL).agg(ITEM_COL).median().rename('median_nunique_items_in_basket').reset_index()
    # user_item_features = user_item_features.merge(__df, on=[USER_COL], how='left')
    
    # кол-ва уникальных категорий товаров в корзине юзера
    df = df.merge(item_features[[ITEM_COL, 'commodity_desc']], on=[ITEM_COL])
    _df = df.groupby([USER_COL, 'basket_id'])['commodity_desc'].nunique().reset_index()
    # mean
    __df = _df.groupby(USER_COL)['commodity_desc'].mean().rename('mean_items_categories_in_basket').reset_index()
    user_item_features = user_item_features.merge(__df, on=[USER_COL])
    # min
    # __df = _df.groupby(USER_COL)['commodity_desc'].min().rename('min_items_categories_in_basket').reset_index()
    # user_item_features = user_item_features.merge(__df, on=[USER_COL])
     # max
    __df = _df.groupby(USER_COL)['commodity_desc'].max().rename('max_items_categories_in_basket').reset_index()
    user_item_features = user_item_features.merge(__df, on=[USER_COL])
    # std 
    __df = _df.groupby(USER_COL)['commodity_desc'].std().rename('std_items_categories_in_basket').reset_index()
    user_item_features = user_item_features.merge(__df, on=[USER_COL])    
    # median
    # __df = _df.groupby(USER_COL)['commodity_desc'].median().rename('median_items_categories_in_basket').reset_index()
    # user_item_features = user_item_features.merge(__df, on=[USER_COL])
    
    # объем продаж относительно товара
    # _df = df.groupby(by=ITEM_COL).agg('sales_value').sum().rename('item_sum_sales_value').reset_index()
    # user_item_features = user_item_features.merge(_df, on=[ITEM_COL], how='left')
    
    # cумма количества относительно товара
    # _df = user_item_features.groupby(by=ITEM_COL).agg('quantity').sum().rename('item_sum_quantity').reset_index()
    # user_item_features = user_item_features.merge(_df, on=[ITEM_COL], how='left')
    
    # средний объем продаж товаров в неделю
    # _df = (df.groupby(by=ITEM_COL).agg('sales_value').sum()/df.week_no.nunique()).rename('avg_sales_value_per_week').reset_index()
    # user_item_features = user_item_features.merge(_df, on=[ITEM_COL], how='left')
    
    # цена товара
    # _df = (df.groupby(by=ITEM_COL).agg('sales_value').sum()/df.groupby(by=ITEM_COL).agg('quantity').sum()).rename('item_price').reset_index()
    # user_item_features = user_item_features.merge(_df, on=[ITEM_COL], how='left')

    # Количество юзеров покупавших данный товар
    # _df = user_item_features.groupby(by=ITEM_COL).agg(USER_COL).count().rename('count_users_bought_item').reset_index()
    # user_item_features = user_item_features.merge(_df, on=[ITEM_COL], how='left')
    
     # эмбеддинги товаров
    recommender = MainRecommender(df)
    _df = recommender.model.item_factors
    n_factors = recommender.model.factors
    ind = list(recommender.id_to_itemid.values())
    _df = pd.DataFrame(_df, index=ind).reset_index()
    _df.columns = [ITEM_COL] + [f'factor_{str(i + 1)}' for i in range(n_factors)]
    user_item_features = user_item_features.merge(_df, on=[ITEM_COL])
    
    # эмбеддинги пользователей
    _df = recommender.model.user_factors
    ind = list(recommender.id_to_userid.values())
    _df = pd.DataFrame(_df, index=ind).reset_index()
    _df.columns = [USER_COL] + [f'user_factor_{str(i + 1)}' for i in range(n_factors)]
    user_item_features = user_item_features.merge(_df, on=[USER_COL])
    
    return user_item_features


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


def set_frequency_rank_coding(df, feature_names: list or str, frequency=True):
    if type(feature_names) is str:
        feature_names = [feature_names]
    for feature_name in feature_names:
        feature_vcount = df[feature_name].value_counts()
        feature_vcount_idx_list = feature_vcount.index.tolist()
        
        new_name = f'{feature_name}_freq' if frequency else f'{feature_name}_rank'
        
        for idx, itm in enumerate(feature_vcount_idx_list):
            df.loc[df[feature_name] == itm, new_name] = feature_vcount[itm] if frequency else idx

    return df


def get_candidates(data_train_matcher, data_train_ranker, N_PREDICT, add_to_ranker):
    
    recommender = MainRecommender(data_train_matcher)
    
    users_matcher = data_train_matcher[USER_COL].values
    users_ranker = data_train_ranker[USER_COL].values
    if add_to_ranker:
        users_ranker += add_to_ranker
    
    current_users = list(set(users_ranker) & set(users_matcher))
    new_users = list(set(users_ranker) - set(users_matcher))
    
    df = pd.DataFrame(users_ranker, columns=[USER_COL])
    df_match_candidates = df[USER_COL].isin(current_users)
    df.loc[df_match_candidates, 'candidates'] = df.loc[df_match_candidates, USER_COL].apply(
        lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
    
    if new_users:
        df_ranker_candidates = df[USER_COL].isin(new_users)
        df.loc[df_ranker_candidates, 'candidates'] = df.loc[df_ranker_candidates, USER_COL].apply(
            lambda x: recommender.overall_top_purchases[:N_PREDICT])
        
    return df


def get_target_ranker(data_train_matcher, data_train_ranker, item_features, user_features, user_item_features, N_PREDICT, add_to_ranker=None):
    
    users_ranker = get_candidates(data_train_matcher, data_train_ranker, N_PREDICT, add_to_ranker)
    
    df = pd.DataFrame({USER_COL: users_ranker[USER_COL].values.repeat(N_PREDICT),
                      ITEM_COL: np.concatenate(users_ranker['candidates'].values)
                      })
    df_ranker_train = data_train_ranker[[USER_COL, ITEM_COL]].copy()
    df_ranker_train['target'] = 1  # тут только покупки 

    df_ranker_train = df.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')
    df_ranker_train['target'].fillna(0, inplace= True)
    
    df_ranker_train = df_ranker_train.merge(item_features, on=ITEM_COL, how='left')
    df_ranker_train = df_ranker_train.merge(user_features, on=USER_COL, how='left')
    df_ranker_train = df_ranker_train.merge(user_item_features, on=[USER_COL, ITEM_COL], how='left')
    
    return df_ranker_train


def set_warm_start(data_train_matcher, data_val_matcher, data_train_ranker, data_val_ranker, test=False):
    # ищем общих пользователей
    common_users = list(set(data_train_matcher.user_id.values)&(set(data_val_matcher.user_id.values))&set(data_val_ranker.user_id.values))
    
    # оставляем общих пользователей
    data_train_matcher = data_train_matcher[data_train_matcher.user_id.isin(common_users)]
    data_val_matcher = data_val_matcher[data_val_matcher.user_id.isin(common_users)]
    data_train_ranker = data_train_ranker[data_train_ranker.user_id.isin(common_users)]
    data_val_ranker = data_val_ranker[data_val_ranker.user_id.isin(common_users)]
    
    if test:
        return data_train_matcher, data_val_matcher, data_val_ranker
    else:
        return data_train_matcher, data_val_matcher, data_train_ranker, data_val_ranker

def preparing_data(data_train_matcher, data_train_ranker, item_features, user_features, user_item_features, USER_COL='user_id', ITEM_COL='item_id', N_PREDICT=50, get_df=True):
    
    
    # взяли пользователей из трейна для ранжирования
    recommender = MainRecommender(data_train_matcher)
    
    df_match_candidates = pd.DataFrame(data_train_ranker[USER_COL].unique())
    df_match_candidates.columns = [USER_COL]
    # собираем кандитатов с первого этапа (matcher)
    df_match_candidates['candidates'] = df_match_candidates[USER_COL].apply(lambda x: recommender.get_own_recommendations(x, N=N_PREDICT))
    
    df_items = df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    df_items.name = ITEM_COL
    df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)
    
    df_ranker_train = data_train_ranker[[USER_COL, ITEM_COL]].copy()
    df_ranker_train['target'] = 1  # тут только покупки 
    
    df_ranker_train = df_match_candidates.merge(df_ranker_train, on=[USER_COL, ITEM_COL], how='left')
    
    df_ranker_train['target'].fillna(0, inplace= True)
    
    df_ranker_train = df_ranker_train.merge(item_features, on=ITEM_COL, how='left')
    df_ranker_train = df_ranker_train.merge(user_features, on=USER_COL, how='left')
    
    df_ranker_train = df_ranker_train.merge(user_item_features, on=[USER_COL, ITEM_COL], how='left')
    
    X_train = df_ranker_train.drop('target', axis=1)
    y_train = df_ranker_train[['target']]
    if get_df:
        return df_ranker_train
    else:
        return X_train, y_train


def postfilter_items(user_id, recommednations):
    pass