#include <Servo.h>

Servo esc1;
Servo esc2;
Servo esc3;
Servo esc4;

void setup() {
  Serial.begin(9600);
  esc1.attach(5);    
  esc1.writeMicroseconds(1500);  
  esc2.attach(6);    
  esc2.writeMicroseconds(1500);  
  esc3.attach(10);    
  esc3.writeMicroseconds(1500);  
  esc4.attach(11);    
  esc4.writeMicroseconds(1500);  
  delay(2000);        
  Serial.println("start");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the command until newline
    handleCommand(command);
  }
}

void handleCommand(String command) {
  if (command == "U") {
    esc1.writeMicroseconds(1650);
    esc3.writeMicroseconds(1650);
  } else if (command == "D") {
    // Handle Down command
    esc1.writeMicroseconds(1350);
    esc3.writeMicroseconds(1350);
  } else if (command == "C") {
    // Handle Center command
    esc1.writeMicroseconds(1500);
    esc2.writeMicroseconds(1500);
    esc3.writeMicroseconds(1500);
    esc4.writeMicroseconds(1500);
  } else if (command == "FL") {
    // Handle Front Left command
    esc2.writeMicroseconds(1500);
    esc4.writeMicroseconds(1350);
  } else if (command == "FR") {
    // Handle Front Right command
    esc2.writeMicroseconds(1650);
    esc4.writeMicroseconds(1500);
  } else if (command == "BL") {
    // Handle Back Left command
    esc2.writeMicroseconds(1500);
    esc4.writeMicroseconds(1650);
  } else if (command == "BR") {
    // Handle Back Right command
    esc2.writeMicroseconds(1350);
    esc4.writeMicroseconds(1500);
  } else if (command == "F") {
    // Handle Forward command
    esc2.writeMicroseconds(1650);
    esc4.writeMicroseconds(1350);
  } else if (command == "B") {
    // Handle Backward command
    esc2.writeMicroseconds(1350);
    esc4.writeMicroseconds(1650);
  } else if (command == "L") {
    // Handle Left command
    esc2.writeMicroseconds(1350);
    esc4.writeMicroseconds(1350);
  } else if (command == "R") {
    // Handle Right command
    esc2.writeMicroseconds(1650);
    esc4.writeMicroseconds(1650);
  } else {
    // Handle unknown command
    esc1.writeMicroseconds(1500);
    esc2.writeMicroseconds(1500);
    esc3.writeMicroseconds(1500);
    esc4.writeMicroseconds(1500);
  }
}
