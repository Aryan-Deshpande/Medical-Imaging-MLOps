from email.policy import default
from click import command
from kubernetes import client as k8s_client
import kfp.dsl as dsl
import kfp.compiler as compiler
import kfp.components as components
import kfp

@dsl.pipeline(
    name="Pytorch Medical Scanning",
    description=" A Machine Learning Pipeline To Train Over Brain Tumour Data that is collected from the Client or Medical Instituition. Constructing a Model that aids in Dectecting Tumours in the Brain."
)

def pipe(
    tenant_id,
    service_principal_id,
    service_principal_password,
    subscription_id,
    resource_group,
    workspace
):

    """ This is the Pipeline """

    persistant_volume_path = '/mnt/azure',
    training_folder = "./training",
    dataset = "dataset_uri",
    mdl = './model',
    learning_rate = 0.01,
    model_name = "brain",
    model_store= "s3uri",
    epochs = 1000,
    batch_size = 10,

    data_download='azure://data',
    data_upload='azure://data',
    operations = dict()

    operations['register'] = dsl.ContainerOp(
        name='register',
        image='insert image name:tag',
        command=['python'],
        arguments=[
        '/scripts/register.py',
        '--base_path', persistant_volume_path,
        '--model', 'latest.h5',
        '--model_name', model_name,
        '--tenant_id', tenant_id,
        '--service_principal_id', service_principal_id,
        '--service_principal_password', service_principal_password,
        '--subscription_id', subscription_id,
        '--resource_group', resource_group,
        '--workspace', workspace
        ]
  )

    operations['preprocess'] = dsl.ContainerOp(
        name='preprocess',
        image='preprocess:latest',
        command=['python3'], arguments=[
            '--base_path', persistant_volume_path,
            '--data', training_folder,
            '--target', dataset,
            '--zipfile', 

        ]
    )

    # fetching the data ?
    operations['fetch'] = dsl.ContainerOp(
        name='fetch',
        image='fetch:latest',
        command=['python3'], arguments=[
        '--base_path',persistant_volume_path,
        '--data', training_folder,
        '--target', dataset]
    )

    operations['fetch'].after(operations['preprocess'])

    operations['train'] = dsl.ContainerOp(
        name='trainging',
        image='train:latest',
        command=['python3'], arguments=[
            '--base_path', persistant_volume_path,
            '--data', training_folder,
            '--outputs', mdl,
        ]
    )

    operations['train'].after(operations['fetch'])

    operations['export'] = dsl.ContainerOp(
        name='export',
        image='export:latest',
        command=['python3'], arguments=[
        '--base_path',persistant_volume_path,
        '--model','./model',
        '--s3buck', model_store ]
    )

    operations['register'].after(operations['train'])
    operations['export'].after(operations['register'])

    # kfserving ? 
    kfserving = components.load_component_from_file(
        '/serve/kfserving-component.yaml'
    )

    operations['serving'] = kfserving(
        actions='apply',
        default_model_uri=f's3_uri',
        modelnm = 'chest',
        framework='pytorch',
    )

    operations['serving'].after(operations['export'])

    for _, op in operations.items():
        op.container.set_image_pull_policy('Always')
        op.add_volume(
            name='azure',
            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                claim_name='azure-disk'
        )
    ).add_volume_mount(k8s_client.V1VolumeMount(
        mount_path='/mnt/azure', name='azure'
    ))

# To create tar.gz file and upload to Kubeflow Pipeline UI
if __name__=='__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile('pipe', __file__ + '.tar.gz')

# Code when run in the notebook
"""client = kfp.Client()
run_result = client.create_run_from_pipeline_func(
    pipeline_func=pipe,
    experiment_name='experiment_name',
    run_name='run_name',
    arguments='arguments',
)"""
    