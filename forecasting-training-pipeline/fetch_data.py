import kfp
import kfp.dsl as dsl
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
)

@component(base_image='python:3.9', packages_to_install=["dask[dataframe]==2024.8.0", "s3fs==2025.2.0", "pandas==2.2.3"])

def fetch_data(
    data_source: dict,
    dataset: Output[Dataset]
):
    """
    Fetches data from URL
    """
    
    import pandas as pd
    import yaml
    import os
    
    song_properties = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_properties.parquet')
    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
    
    data = song_rankings.merge(song_properties, on='spotify_id', how='left')
    
    dataset.path += ".csv"
    dataset.metadata = data_source
    data.to_csv(dataset.path, index=False, header=True)


@component(base_image='python:3.9', packages_to_install=["dvc[s3]==3.1.0", "pathspec<0.12.0", "dask[dataframe]==2024.8.0", "s3fs==2025.2.0", "pandas==2.2.3", "PyYAML==6.0.2"])

def fetch_data_from_dvc(
    dataset: Output[Dataset],
    model_config: dict,
    cluster_domain: str = "",
    git_version: str = "main",
):
    """
    Fetches data from DVC using the new architecture with separate data repo
    """
    
    import pandas as pd
    import yaml
    import json
    import configparser
    import os
    import subprocess
    
    def run_command(command, cwd=None, env=None):
        result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
        return result.stdout.strip()
        
    def read_hash(dvc_file_path):
        with open(dvc_file_path, 'r') as file:
            dvc_data = yaml.safe_load(file)
            md5_hash = dvc_data['outs'][0]['md5']
        return md5_hash

    git_username = os.environ.get('username')
    git_password = os.environ.get('password')
    current_path = os.environ.get("PATH", "")
    new_path = f"{current_path}:/.local/bin"
    os.environ["PATH"] = new_path
    
    print("Updated PATH:", os.environ["PATH"])

    # Extract data source configuration from model config
    data_source = model_config.get('data_source', {})
    if data_source.get('type') != 'dvc':
        raise ValueError("Model data_source type must be 'dvc'")
    
    dataset_name = data_source.get('dataset', 'song_properties.parquet')
    expected_hash = data_source.get('dvc_hash')
    data_repo_url = data_source.get('repo_url', 'https://github.com/nine-thousand-models/nine-thousand-data')
    
    print(f"Fetching dataset: {dataset_name}")
    print(f"Expected DVC hash: {expected_hash}")

    os.chdir("/tmp")

    # Clone the data repository
    run_command(f"git clone https://{git_username}:{git_password}@{data_repo_url.replace('https://', '')} data-repo")
    os.chdir("/tmp/data-repo")
    
    try:
        run_command(f"git checkout {git_version}")
    except Exception as e:
        print(e)
        print(f"Could not check out version {git_version}, using main")

    # Pull the specific dataset using DVC with new folder structure
    dataset_path = dataset_name  # e.g., "datasets/demand-forecaster-data/data.parquet"
    dvc_file_path = f"{dataset_name}.dvc"  # e.g., "datasets/demand-forecaster-data/data.parquet.dvc"
    
    if not os.path.exists(dvc_file_path):
        raise FileNotFoundError(f"DVC file not found: {dvc_file_path}")
    
    # Verify hash matches expected
    actual_hash = read_hash(dvc_file_path)
    if expected_hash and actual_hash != expected_hash:
        print(f"WARNING: Hash mismatch! Expected: {expected_hash}, Actual: {actual_hash}")
    
    # Pull the data file
    run_command(f"dvc pull {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found after DVC pull: {dataset_path}")

    # Read DVC config for metadata
    config = configparser.ConfigParser()
    config.read('.dvc/config')

    # Load the data - assuming parquet format, but could be made configurable
    if dataset_path.endswith('.parquet'):
        main_data = pd.read_parquet(dataset_path)
    elif dataset_path.endswith('.csv'):
        main_data = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    # For backward compatibility, still merge with song_rankings if needed
    # This could be made configurable based on model requirements
    try:
        song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
        if 'spotify_id' in main_data.columns and 'spotify_id' in song_rankings.columns:
            data = song_rankings.merge(main_data, on='spotify_id', how='left')
        else:
            data = main_data
    except Exception as e:
        print(f"Could not merge with song_rankings, using main dataset only: {e}")
        data = main_data
    
    # Save the dataset
    dataset.path += ".csv"
    dataset.metadata = {
        "DVC training data hash": actual_hash,
        "dataset_name": dataset_name,
        "data_repo_url": data_repo_url
    } | {section: str(dict(config.items(section))) for section in config.sections()}
    
    data.to_csv(dataset.path, index=False, header=True)
    print(f"Dataset saved with {len(data)} rows, DVC hash: {actual_hash}")


@component(base_image='python:3.12', packages_to_install=["feast==0.59.0", "psycopg2-binary>=2.9", "dask-expr==1.1.10", "s3fs==2024.6.1", "psycopg_pool==3.2.3", "psycopg==3.2.3", "pandas==2.2.3"])
def fetch_data_from_feast(
    version: str,
    dataset: Output[Dataset]
):
    """
    Fetches data from Feast
    """
    
    import feast
    import pandas as pd
    import numpy as np

    fs_config_json = {
        'project': 'music',
        'provider': 'local',
        'registry': {
            'registry_type': 'sql',
            'path': 'postgresql://feast:feast@feast:5432/feast',
            'cache_ttl_seconds': 60,
            'sqlalchemy_config_kwargs': {
                'echo': False, 
                'pool_pre_ping': True
            }
        },
        'online_store': {
            'type': 'postgres',
            'host': 'feast',
            'port': 5432,
            'database': 'feast',
            'db_schema': 'feast',
            'user': 'feast',
            'password': 'feast'
        },
        'offline_store': {'type': 'file'},
        'entity_key_serialization_version': 2
    }

    fs_config = feast.repo_config.RepoConfig(**fs_config_json)
    fs = feast.FeatureStore(config=fs_config)

    song_rankings = pd.read_parquet('https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet')
    # Feast will remove rows with identical id and date so we add a small delta to each
    microsecond_deltas = np.arange(0, len(song_rankings))*2
    song_rankings['snapshot_date'] = pd.to_datetime(song_rankings['snapshot_date'])
    song_rankings['snapshot_date'] = song_rankings['snapshot_date'] + pd.to_timedelta(microsecond_deltas, unit='us')

    feature_service = fs.get_feature_service("serving_fs")

    data = fs.get_historical_features(entity_df=song_rankings, features=feature_service).to_df()

    features = [f.name for f in feature_service.feature_view_projections[0].features]
    
    dataset.metadata = {"song_properties": "serving_fs", "song_rankings": "https://github.com/rhoai-mlops/jukebox/raw/refs/heads/main/99-data_prep/song_rankings.parquet", "features": features}
    dataset.path += ".csv"
    data.to_csv(dataset.path, index=False, header=True)