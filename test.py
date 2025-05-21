from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="automax_geocoder")
location = geolocator.reverse("44.2540, -76.5675")
print(location.address)
print((location.latitude, location.longitude))
(40.7410861, -73.9896297241625)
print(location.raw)
