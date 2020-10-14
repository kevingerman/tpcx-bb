from xbb_tools.utils import benchmark, tpcxbb_argparser, run_query
from xbb_tools.readers import build_reader
import os, subprocess, math, time
from xbb_tools.config import get_config

# these tables have extra data produced by bigbench dataGen
refresh_tables = [
    "customer",
    "customer_address",
    "inventory",
    "item",
    "item_marketprices",
    "product_reviews",
    "store_returns",
    "store_sales",
    "web_clickstreams",
    "web_returns",
    "web_sales",
]

def get_tables( spark_schema_dir ):
    return [table.split(".")[0] for table in os.listdir(spark_schema_dir)]

# Spark uses different names for column types, and RAPIDS doesn't yet support Decimal types.
def get_schema(table, schema_dir):
    with open(os.path.join(schema_dir,f"{table}.schema")) as fp:
        schema = fp.read()
        names = [line.replace(",", "").split()[0] for line in schema.split("\n")]
        types = [
            line.replace(",", "")
            .split()[1]
            .replace("bigint", "int")
            .replace("string", "str")
            for line in schema.split("\n")
        ]
        types = [
            col_type.split("(")[0].replace("decimal", "float") for col_type in types
        ]
        return names, types


def read_csv_table(table, data_dir, schema_dir, chunksize="256 MiB"):
    names, types = get_schema(table, schema_dir)
    dtype=dict(zip(names, types))
    base_path = f"{data_dir}/data/{table}"
    files = os.listdir(base_path)
    # item_marketprices has "audit" files that should be excluded
    if table == "item_marketprices":
        paths = [
            f"{base_path}/{fn}"
            for fn in files
            if "audit" not in fn and os.path.getsize(f"{base_path}/{fn}") > 0
        ]
        base_path = f"{data_dir}/data_refresh/{table}"
        paths = paths + [
            f"{base_path}/{fn}"
            for fn in os.listdir(base_path)
            if "audit" not in fn and os.path.getsize(f"{base_path}/{fn}") > 0
        ]
        df = dask_cudf.read_csv(
            paths, sep="|", names=names, dtype=dtype, chunksize=chunksize, quoting=3
        )
    else:
        paths = [
            f"{base_path}/{fn}"
            for fn in files
            if os.path.getsize(f"{base_path}/{fn}") > 0
        ]
        if table in refresh_tables:
            base_path = f"{data_dir}/data_refresh/{table}"
            paths = paths + [
                f"{base_path}/{fn}"
                for fn in os.listdir(base_path)
                if os.path.getsize(f"{base_path}/{fn}") > 0
            ]
        df = dask_cudf.read_csv(
            paths, sep="|", names=names, dtype=types, chunksize=chunksize, quoting=3
        )

    return df


def multiplier(unit):
    if unit == "G":
        return 1
    elif unit == "T":
        return 1000
    else:
        return 0


# we use size of the CSV data on disk to determine number of Parquet partitions
def get_size_gb(table, data_dir):
    table_path=os.path.join( data_dir, 'data', table)
    print( f"Getting size of {table_path}")
    size = subprocess.check_output(["du", "-sh", table_path]).split()[0].decode("utf-8")
    unit = size[-1]

    size = math.ceil(float(size[:-1])) * multiplier(unit)

    if table in refresh_tables:
        table_path = os.path.join(data_dir,'data_refresh',table)
        refresh_size = (
            subprocess.check_output(["du", "-sh", table_path]).split()[0].decode("utf-8")
        )
        size = size + math.ceil(float(refresh_size[:-1])) * multiplier(refresh_size[-1])

    return size


def repartition(table, data_dir, schema_dir, outdir, npartitions=None, chunksize=None, compression="snappy"):
    size = get_size_gb(table, data_dir)
    if npartitions is None:
        npartitions = max(1, size)

    print(
        f"Converting {table} of {size} GB to {npartitions} parquet files, chunksize: {chunksize}"
    )
    # web_clickstreams is particularly memory intensive
    # we sacrifice a bit of speed for stability, converting half at a time
    if table in ["web_clickstreams"]:
        df = read_csv_table(table, data_dir, schema_dir, chunksize)
        half = max(1,int(df.npartitions / 2))
        df.partitions[0:half].repartition(npartitions=max(1,int(npartitions / 2))).to_parquet(
            outdir + table, compression=compression
        )
        print("Completed first half of web_clickstreams..")
        df.partitions[half:].repartition(npartitions=max(1,int(npartitions / 2))).to_parquet(
            outdir + table, compression=compression
        )

    else:
        read_csv_table(table, data_dir, schema_dir, chunksize).repartition(
            npartitions=npartitions
        ).to_parquet(outdir + table, compression=compression)


def main(client, config):
    # location you want to write Parquet versions of the table data
    data_dir = config.get("data_dir",'.')
    outdir = f"{config.get('output_dir',data_dir)}/parquet_{config.get('partitions',3)}gb/"
    schema_dir= config.get('spark_schema_dir',
                           os.path.join(os.getcwd(),'..','..','spark_table_schemas'))
    began = time.time()
    for table in get_tables(schema_dir):
        size_gb = get_size_gb(table, data_dir)
        # product_reviews has lengthy strings which exceed cudf's max number of characters per column
        # we use smaller partitions to avoid overflowing this character limit
        if table == "product_reviews":
            npartitions = max(1, int(size_gb / 1))
        else:
            npartitions = max(1, int(size_gb / config.get('partitions',3)))
        repartition(table, data_dir, schema_dir, outdir, npartitions, config.get('chunk_size',"128 MiB"), compression="snappy")
    print(f"{config.get('chunk_size','128 MiB')} took {time.time() - began}s")
    return cudf.DataFrame()


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    conf = tpcxbb_argparser()
    client, bc = attach_to_cluster(conf)
    run_query(config=conf, client=client, query_func=main)
