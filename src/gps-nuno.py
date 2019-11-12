
#Script to read USB GPS data directly from port - by Nuno Vieira
#Pyserial must be installed first


import serial

with serial.Serial('/dev/ttyUSB0', 4800, timeout=1) as ser:
	latitude = ''
	longitude = ''
	while True:
		line = ser.readline().decode('ascii', errors='replace')
		if "GPGGA" in line.strip():
			print(line.strip())
			time,lat,dir1,longi,dir2,qual,sats = line.strip().split(",")[1:8]

			print("UTC time: "+time[0:2]+":"+time[2:4]+":"+time[4:6])
			print("Latitude: " + lat + " " + dir1)
			print("Longitude: " + longi + " " + dir2)
			print("Used satellites: " + sats)
			print()
		    
#1    = UTC of Position
#2    = Latitude
#3    = N or S
#4    = Longitude
#5    = E or W
#6    = GPS quality indicator (0=invalid; 1=GPS fix; 2=Diff. GPS fix)
#7    = Number of satellites in use [not those in view]
#8    = Horizontal dilution of position
#9    = Antenna altitude above/below mean sea level (geoid)
#10   = Meters  (Antenna height unit)
#11   = Geoidal separation (Diff. between WGS-84 earth ellipsoid and
#       mean sea level.  -=geoid is below WGS-84 ellipsoid)
#12   = Meters  (Units of geoidal separation)
#13   = Age in seconds since last update from diff. reference station
#14   = Diff. reference station ID#
#15   = Checksum
