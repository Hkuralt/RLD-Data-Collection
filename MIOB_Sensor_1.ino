/*

*/
#include <Arduino_LSM9DS1.h>
float accx, accy, accz, gyx, gyy, gyz, magx, magy, magz;

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }  
    IMU.setAccelODR(2);
    IMU.setGyroODR(2);
    //IMU.setMagnetODR(6);
}

void loop() {
  if ((IMU.accelerationAvailable())&&(IMU.gyroscopeAvailable())&&(IMU.magneticFieldAvailable())) {
    IMU.readAcceleration(accx, accy, accz);
    IMU.readGyroscope(gyx, gyy, gyz);
    IMU.readMagneticField(magx, magy, magz);
    
    Serial.print(accx);
    Serial.print(' ');
    Serial.print(accy);
    Serial.print(' ');
    Serial.print(accz);
    Serial.print(' ');
    
    Serial.print(gyx);
    Serial.print(' ');
    Serial.print(gyy);
    Serial.print(' ');
    Serial.print(gyz);
    Serial.print(' ');
    
    Serial.print(magx);
    Serial.print(' ');
    Serial.print(magy);
    Serial.print(' ');
    Serial.println(magz);
    delay(20);
  }
}
