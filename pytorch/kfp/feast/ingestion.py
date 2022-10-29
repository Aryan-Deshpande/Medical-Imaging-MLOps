from feast import Client
from feast import Entity, ValueType, Feature, FileSource

# Connect to a feast deployement
Client = Client(core_url='localhost:6565', serving_url='localhost:6566')
# client is then connected

# creating an entity is done by creating an Entity object
id = Entity(name="id", description="cluster_id", value_type=ValueType.INT64, labels={"team": "cluster"})
Client.apply(id)

# creating a feature is done by creating a Feature object
# and then applying it to the client

# create a feature
feature = Feature(name="images", entities=["id"],
features=[
    Feature("img", dtype=ValueType.BYTES, entities=["id"])],
    
    batch_source = FileSource(
    file_format = "parquet",
    file_url = "file:///home/aryan/ingestion/",
    event_timestamp_column = "event_timestamp",
    created_timestamp_column = "created_timestamp",

), 
    )
Client.apply(feature)
# now the feature is applied to the client

# now we can ingest the data
Client.ingest(feature, max_workers=10, timeout=20000)

# now we can check the status of the ingestion
Client.get_feature_set(name="images", project="default")
