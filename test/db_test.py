import json
from pymongo import MongoClient as Client

def get_db_data(country, season):

  myclient = Client('mongodb://localhost:27017/')
  mydb = myclient['bs_db']
  mycol = mydb[country]
  countQuery = {'season': season}
  count = mycol.find(countQuery).count()
  dataQuery = {'season': season, 'week': str(count)}
  for x in mycol.find(dataQuery):
    x.pop('_id')
    print(x)


if __name__ == '__main__':
  get_db_data('england', '2005-06')