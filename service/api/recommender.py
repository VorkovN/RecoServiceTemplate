import pickle

import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset


class Recommender:
    def __init__(self, dataset_path: str, warm_model_path: str, hot_model_path: str, offline_rec_flag: bool):
        # Десереализуем датасет
        self.offline_rec_flag = offline_rec_flag
        self.interactions = pd.read_csv(dataset_path)
        self.interactions.rename(
            columns={"last_watch_dt": Columns.Datetime,
                     "total_dur": Columns.Weight}, inplace=True)
        self.dataset = Dataset.construct(self.interactions)

        # Выбираем самое популярное
        items_ids_all = self.interactions.groupby(Columns.Item)[Columns.User].nunique().reset_index(name="unique_users_count")
        self.popular_items = items_ids_all.sort_values(by="unique_users_count", ascending=False).head(10)[Columns.Item]

        # Запоминаем отсутствующих юзеров
        self.missing_user_id_values = set(range(1100000)).difference(set(self.interactions[Columns.User]))

        # Сохраняем список горячих юзеров
        user_ids_all = self.interactions.groupby(Columns.User)[Columns.Item].nunique().reset_index(name="unique_items_count")
        self.hot_users = user_ids_all[user_ids_all["unique_items_count"] > 20][Columns.User]
        print(f"Hot users cout: {self.hot_users.shape[0]}")

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
        if self.hot_users.isin([user_id]).any():
            user_id_kostyl = pd.DataFrame({Columns.User: [user_id]})

            if self.offline_rec_flag == True:
                recos = self.recos_hot[self.recos_hot[Columns.User].isin([user_id])][Columns.Item]
            else:
                recos = self.hot_model.predict(user_id_kostyl)

            print(f"user_id {user_id} is hot; recos {recos}; len{len(recos)}")
            return recos[Columns.Item]

        # Теплый
        if user_id not in self.missing_user_id_values:
            if self.offline_rec_flag == True:
                recos = self.warm_model.recommend(users=[user_id], dataset=self.dataset, k=k_recs, filter_viewed=True)
                recos = self.recos_warm[self.recos_warm[Columns.User].isin([user_id])][Columns.Item]
            else:
                recos = self.popular_items

            print(f"user_id {user_id} is warm; recos {recos}; len{len(recos)}")
            return recos

        # Холодный
        recos = self.popular_items

        print(f"user_id {user_id} is cold; recos {recos}; len{len(recos)}")
        return recos
