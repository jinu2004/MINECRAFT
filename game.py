import pygame
import sys
import speech_recognition as sr

# Initialize Pygame
pygame.init()

# Set up display
width, height = 400, 300
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Voice Controlled Pygame Example")

# Set up the box
box_size = 50
box_x = (width - box_size) // 2
box_y = (height - box_size) // 2
box_speed = 20

# Initialize speech recognition
recognizer = sr.Recognizer()

# Main game loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Listen for voice input
    with sr.Microphone() as source:
        print("Say 'forward', 'downward', 'left', or 'right'")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Recognize voice command
        command = recognizer.recognize_google(audio).lower()
        
        # Update box position based on voice input
        if "forward" in command:
            box_y -= box_speed
        elif "downward" in command:
            box_y += box_speed
        elif "left" in command:
            box_x -= box_speed
        elif "right" in command:
            box_x += box_speed

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    # Draw background and box
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0, 0, 255), (box_x, box_y, box_size, box_size))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)
