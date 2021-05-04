from periphery import GPIO
import time

class GPIO_Module:
    def __init__(self):
        self.push_button = GPIO("/dev/gpiochip0",6,"in") # Pin 13
        self.led_col0 = GPIO("/dev/gpiochip2",9,"out") # Pin 16
        self.led_col1 = GPIO("/dev/gpiochip4",10,"out") # Pin 18
        self.led_col2 = GPIO("/dev/gpiochip4",12,"out") # Pin 22
        self.led_col3 = GPIO("/dev/gpiochip0",7,"out") # Pin 29
        self.led_col4 = GPIO("/dev/gpiochip0",8,"out") # Pin 31
        self.led_col5 = GPIO("/dev/gpiochip4",13,"out") # Pin 36
        self.led_col6 = GPIO("/dev/gpiochip2",13,"out") # Pin 37

        self.led_list = [self.led_col0,self.led_col1,self.led_col2,self.led_col3,self.led_col4,self.led_col5,self.led_col6]
        self.off_all_led()

    def wait_push(self) :
        while(not self.push_button.read()) :
            self.push_button.read()
        #push_button.close()
        print('pushed')

    def on_all_led(self):
        for i in range(7):
            self.led_list[i].write(True)

    def off_all_led(self):
        for i in range(7):
            self.led_list[i].write(False)

    def on_led(self,col) :
        self.led_list[col].write(True)


    def off_led(self,col) :
        self.led_list[col].write(False)
        #led_list[col].close()
    
    def showWinner(self,winner):
        if winner == 1 :
            led = self.led_col0
        elif winner == 2:
            led = self.led_col6
        else:
            led = self.led_col3
        self.on_all_led()
        for i in range(20):
            led.write(True)
            time.sleep(1)
            led.write(False)
            time.sleep(1)
            
    def showConfirmButton(self):
        self.on_all_led()
        time.sleep(1)
        self.off_all_led()


    # Test code
    # while(True) :
    #     wait_push()
    #     for index_col in range(len(led_list)) :
    #         print('LED '+str(index_col))
    #         on_led(index_col)
    #         time.sleep(1)
    #         off_led(index_col)
    #         time.sleep(1)
