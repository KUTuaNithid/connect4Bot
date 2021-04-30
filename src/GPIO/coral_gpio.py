from periphery import GPIO
import time

push_button = GPIO("/dev/gpiochip0",6,"in") # Pin 13
led_col0 = GPIO("/dev/gpiochip2",9,"out") # Pin 16
led_col1 = GPIO("/dev/gpiochip4",10,"out") # Pin 18
led_col2 = GPIO("/dev/gpiochip4",12,"out") # Pin 22
led_col3 = GPIO("/dev/gpiochip0",7,"out") # Pin 29
led_col4 = GPIO("/dev/gpiochip0",8,"out") # Pin 31
led_col5 = GPIO("/dev/gpiochip4",13,"out") # Pin 36
led_col6 = GPIO("/dev/gpiochip2",13,"out") # Pin 37

led_list = [led_col0,led_col1,led_col2,led_col3,led_col4,led_col5,led_col6]


def wait_push() :
    while(not push_button.read()) :
        push_button.read()
        print('wait')
    #push_button.close()
    print('push')


def on_led(col) :
    led_list[col].write(True)


def off_led(col) :
    led_list[col].write(False)
    #led_list[col].close()


# Test code
while(True) :
    wait_push()
    for index_col in range(len(led_list)) :
        print('LED '+str(index_col))
        on_led(index_col)
        time.sleep(1)
        off_led(index_col)
        time.sleep(1)
