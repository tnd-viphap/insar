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

if __name__ == "__main__":
    wkt_string = "POLYGON((106.6969 10.7615,106.7275 10.7615,106.7275 10.7945,106.6969 10.7945,106.6969 10.7615))"
    filename = "demo.shp"
    converter = GeomConverter()
    converter.wkttoshp(wkt_string, filename)
