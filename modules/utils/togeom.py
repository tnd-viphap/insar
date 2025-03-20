import geopandas as gpd
from shapely.wkt import loads

class GeomConverter:
    def __init__(self):
        super().__init__()
    
    def wkttoshp(self, wkt_string, filename):
        geometry = loads(wkt_string)
        gdf = gpd.GeoDataFrame({"geometry": [geometry]})
        gdf.set_crs(epsg=4326, inplace=True)
        gdf.to_file(f"./geom/{filename}")
        
    def wkt_to_bbox(self, wkt_string):
        try:
            # Parse the WKT string into a Shapely geometry
            geometry = loads(wkt_string)
            # Get the bounding box
            bbox = geometry.bounds
            return list(bbox)
        except Exception as e:
            print(f"Error converting WKT to bbox: {e}")
            return None
        
    def bboxtowkt(self, bbox):
        return f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"

if __name__ == "__main__":
    #wkt_string = "POLYGON((108.8721 15.1294,108.8996 15.1294,108.8996 15.1569,108.8721 15.1569,108.8721 15.1294))"
    bbox = [106.691059, 20.837039, 106.7762203, 20.899435]
    converter = GeomConverter()
    print(converter.bboxtowkt(bbox))
    #print(converter.wkt_to_bbox(wkt_string))