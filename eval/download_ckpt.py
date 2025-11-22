import tinker
import urllib.request
 
sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path("tinker://9a511290-aa5e-5879-b111-5a76dfb2d179:train:0/sampler_weights/000001")
checkpoint_archive_url_response = future.result()
 
# `checkpoint_archive_url_response.url` is a signed URL that can be downloaded
# until checkpoint_archive_url_response.expires
urllib.request.urlretrieve(checkpoint_archive_url_response.url, "archive.tar")

import tarfile
with tarfile.open("archive.tar", "r") as tar:
    tar.extractall("archive")