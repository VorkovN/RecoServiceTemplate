import pickle

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from service.api.constants import MAX_USER_COUNT

class Recommender:
    def __init__(self, dataset_path: str, users_path: str, items_path: str, warm_model_path: str, hot_model_path: str, offline_rec_flag: bool):
        # Десереализуем датасет
        self.offline_rec_flag = offline_rec_flag
        users = pd.read_csv(users_path)
        items = pd.read_csv(items_path)
        self.interactions = pd.read_csv(dataset_path)
        self.interactions.rename(
            columns={"last_watch_dt": Columns.Datetime,
                     "total_dur": Columns.Weight}, inplace=True)
        self.dataset = Dataset.construct(self.interactions)

        # Выбираем самое популярное
        items_ids_all = self.interactions.groupby(Columns.Item)[Columns.User].nunique().reset_index(name="unique_users_count")
        self.popular_items = items_ids_all.sort_values(by="unique_users_count", ascending=False).head(10)[Columns.Item]

        # Запоминаем отсутствующих юзеров
        self.missing_user_id_values = set(range(MAX_USER_COUNT)).difference(set(self.interactions[Columns.User]))

        # Сохраняем список горячих юзеров
        interactions_df = self.interactions

        user_ids_all = interactions_df.groupby(Columns.User)[
            Columns.Item].nunique().reset_index(name='unique_items_count')
        hot_users = user_ids_all[user_ids_all['unique_items_count'] > 5][
            Columns.User]
        interactions_df_hot_users = interactions_df[
            interactions_df[Columns.User].isin(hot_users)]

        users = users[
            users[Columns.User].isin(interactions_df_hot_users[Columns.User])]
        interactions_df_hot_users = interactions_df_hot_users[
            interactions_df_hot_users[Columns.User].isin(users[Columns.User])]
        items = items[
            items[Columns.Item].isin(interactions_df_hot_users[Columns.Item])]

        self.interactions = interactions_df_hot_users

        user_features_frames = []
        for feature in ["sex", "age", "income"]:
            feature_frame = users.reindex(columns=[Columns.User, feature])
            feature_frame.columns = ["id", "value"]
            feature_frame["feature"] = feature
            user_features_frames.append(feature_frame)
        user_features = pd.concat(user_features_frames)
        user_features.head()

        items["genre"] = items["genres"].str.lower().str.replace(", ", ",",regex=False).str.split(",")
        genre_feature = items[["item_id", "genre"]].explode("genre")
        genre_feature.columns = ["id", "value"]
        genre_feature["feature"] = "genre"
        genre_feature.head()

        content_feature = items.reindex(columns=[Columns.Item, "content_type"])
        content_feature.columns = ["id", "value"]
        content_feature["feature"] = "content_type"

        item_features = pd.concat((genre_feature, content_feature))

        self.dataset = Dataset.construct(
            interactions_df=self.interactions,
            user_features_df=user_features,
            cat_user_features=["sex", "age", "income"],
            item_features_df=item_features,
            cat_item_features=["genre", "content_type"],
        )
        # self.dataset = Dataset.construct(self.interactions)
        print(f"interactions cout: {self.interactions.shape[0]}")

        # Десереализуем холодную модель
        with open(warm_model_path, "rb") as file:
            self.warm_model = pickle.load(file)

        # Десереализуем горячую модель
        with open(hot_model_path, "rb") as file:
            self.hot_model = pickle.load(file)

        print("Models loaded")

        if self.offline_rec_flag == True:
            df_hot = pd.DataFrame({Columns.User:
            self.interactions[self.interactions[Columns.User].isin(self.hot_users)][Columns.User]})
            self.recos_hot = self.hot_model.predict(df_hot)

            print("Hot recos predicted")

            df_warm = self.interactions[
            ~self.interactions[Columns.User].isin(df_hot[Columns.User])].drop_duplicates(subset=Columns.User)
            self.recos_warm = self.warm_model.recommend(
                users=df_warm[Columns.User],
                dataset=self.dataset,
                k=10,
                filter_viewed=True,
            )

            print("Warm recos predicted")

            self.recos_cold = self.popular_items

            print("Cold recos predicted")

    def recommend(self, user_id: int, k_recs: int):
        # Горячий
        if self.interactions[Columns.User].isin([user_id]).any():

            if self.offline_rec_flag == True:
                recos = self.recos_hot[self.recos_hot[Columns.User].isin([user_id])][Columns.Item]
            else:
                print(user_id)
                recos = self.hot_model.recommend(users=[user_id], dataset=self.dataset, k=k_recs, filter_viewed=True)

            # print(f"user_id {user_id} is hot; recos {recos}")
            return recos[Columns.Item]

        # Теплый
        if user_id not in self.missing_user_id_values:
            if self.offline_rec_flag == True:
                recos = self.recos_warm[self.recos_warm[Columns.User].isin([user_id])][Columns.Item]
            else:
                recos = self.popular_items

            # print(f"user_id {user_id} is warm")
            return recos

        # Холодный
        recos = self.popular_items

        # print(f"user_id {user_id} is cold")
        return recos
