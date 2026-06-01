import requests

GEOSERVER  = "http://localhost:8080/geoserver"
WORKSPACE  = "prithvi"
AUTH       = ('admin', 'geoserver')
HEADERS_XML = {"Content-Type": "application/xml"}

def ensure_workspace():
    r = requests.get(
        f"{GEOSERVER}/rest/workspaces/{WORKSPACE}",
        auth=AUTH
    )
    if r.status_code == 404:
        requests.post(
            f"{GEOSERVER}/rest/workspaces",
            auth=AUTH,
            headers=HEADERS_XML,
            data=f"<workspace><name>{WORKSPACE}</name></workspace>"
        )

def publish_cog(layer_name, cog_path):
    ensure_workspace()

    # Create store
    store_xml = f"""
    <coverageStore>
        <name>{layer_name}</name>
        <type>GeoTIFF</type>
        <enabled>true</enabled>
        <workspace>{WORKSPACE}</workspace>
        <url>file:{cog_path}</url>
    </coverageStore>
    """
    requests.post(
        f"{GEOSERVER}/rest/workspaces/{WORKSPACE}/coveragestores",
        auth=AUTH,
        headers=HEADERS_XML,
        data=store_xml
    )

    # Publish layer
    coverage_xml = f"""
    <coverage>
        <name>{layer_name}</name>
        <title>{layer_name}</title>
        <defaultInterpolationMethod>
            <name>nearest neighbor</name>
        </defaultInterpolationMethod>
    </coverage>
    """
    r = requests.post(
        f"{GEOSERVER}/rest/workspaces/{WORKSPACE}/coveragestores/{layer_name}/coverages",
        auth=AUTH,
        headers=HEADERS_XML,
        data=coverage_xml
    )

    # Apply segmentation style
    style_xml = f"""
    <layer>
        <defaultStyle>
            <name>segmentation_style</name>
            <workspace>{WORKSPACE}</workspace>
        </defaultStyle>
    </layer>
    """
    requests.put(
        f"{GEOSERVER}/rest/layers/{WORKSPACE}:{layer_name}",
        auth=AUTH,
        headers=HEADERS_XML,
        data=style_xml
    )

    wms_url = (
        f"{GEOSERVER}/{WORKSPACE}/wms?SERVICE=WMS&VERSION=1.1.1"
        f"&REQUEST=GetMap&LAYERS={WORKSPACE}:{layer_name}"
        f"&WIDTH=800&HEIGHT=800&SRS=EPSG:32642&FORMAT=image/png"
    )
    return wms_url

def delete_layer(layer_name):
    requests.delete(
        f"{GEOSERVER}/rest/workspaces/{WORKSPACE}/coveragestores/{layer_name}?recurse=true",
        auth=AUTH
    )