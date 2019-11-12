
#Script to read USB GPS data directly from port - by Nuno Vieira
#Generator feature added by Marius Fersigan
#Pyserial must be installed first


import serial
import threading
import logging

class GPS:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0', 4800, timeout=1)
        self.longitude = 0.0
        self.latitude = 0.0
        self.time = ''
        self.run = True
        self.thread = threading.Thread(target=self._start, daemon=True)

    def _start(self):
        supress_warn = 0
        while self.run:
            try:
                line = self.ser.readline().decode('ascii', errors='replace').strip()
                if "GPGGA" in line:
                    time,lat,_,longi,_,qual,n_sats = line.split(",")[1:8]
                    if not int(qual):
                        if not supress_warn:
                            logging.warning('GPS: poor signal: %s', line)
                        supress_warn = (supress_warn + 1) % 30
                        longi, lat = 0, 0
                    else:
                        supress_warn = 0
                    self.longitude = float(longi)
                    self.latitude = float(lat)
                    self.time = time[0:2] + ":" + time[2:4] + ":" + time[4:6]
            except:
                logging.warning("GPS might be disconnected.. please check the connection.")

    def start(self):
        self.thread.start()
    
    def stop(self):
        self.run = False
        self.thread._stop()
    
    def get_position(self):
        current_position = { "long": self.longitude, "lat": self.latitude, "time": self.time}
        #print(current_position)
        return(current_position)

    def __del__(self):
        self.stop()




    


                


# with serial.Serial('/dev/ttyUSB0', 4800, timeout=1) as ser:
# 	latitude = ''
# 	longitude = ''
# 	while True:
# 		line = ser.readline().decode('ascii', errors='replace')
# 		if "GPGGA" in line.strip():
# 			print(line.strip())
# 			time,lat,dir1,longi,dir2,qual,sats = line.strip().split(",")[1:8]

# 			print("UTC time: "+time[0:2]+":"+time[2:4]+":"+time[4:6])
# 			print("Latitude: " + lat + " " + dir1)
# 			print("Longitude: " + longi + " " + dir2)
# 			print("Used satellites: " + sats)
# 			print()
		    
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
