from pymongo import MongoClient
import pandas as pd
import json


# we define a class to connect our python script to MongoDb Database deployed on cloud,
# the we load data to the database and retrive data from database
class MongoDbAtlas:

    # this instance method will be used to fetch credentials from a Json file to autnenticate and connect to the database\
    def fetch_creds(self, Json):
        # loadinhg contents from JSON
        with open(Json, "r") as handler:
            contents = json.load(handler)
        return contents["Username"], contents["Password"], contents["host_address"]

    # we create a instance class for create MongoDB client to connect with the database
    def connect_database(self, username, password, host_address):
        url = f"mongodb+srv://{username}:{password}@{host_address}"
        self.client = MongoClient(url)

    # this instance method will create a database if it doesnt exists or access the database if it exist.
    # then we will create a collection
    def create_database(self, database_name, collection_name):
        # database: multiple ollection stored in database
        db = self.client[database_name]
        # collection: multiple documents makes up a collection
        colc = db[collection_name]
        return colc

    # def print_abc(self):
    # print("abc")

    # we define a method to store the stok market data (which is in a dataframe structure) into the mongodb's database's collection
    def store_stockdata(self, stock_df, colc):
        # mongodb store records in BSON which is s a binary-encoded serialization of JSON documents
        # JSON-javascript oject notation-dictionary
        # so we convert the datafrae into a dictionary so that it can be stored into MongoDb database as documents
        stock_dict = stock_df.to_dict("records")
        # display(stock_df)
        # display(stock_dict)
        # we use mongodb's nosqlinsert_many function to insert the dictionary into the mongodb database
        colc.insert_many(stock_dict)

    def fetch_dbdata(self, colc):
        # this method is used to extract the documents of the collection stored in the database
        # we use the find() function to find all possible documents in the database
        dict_fromdb = colc.find()
        dict_list = []
        # all the individual documents will be appended in a list so that we have a list of individual documents
        for x in dict_fromdb:
            dict_list.append(x)
        # display(dict_list)
        df_fromdb = pd.DataFrame.from_dict(dict_list)
        # display(df_fromdb)
        return dict_list, df_fromdb




