import msvcrt  # For Windows systems
import os
import json
from datetime import datetime
import logging
from colorama import init, Fore, Style

class InteractiveRefinement:
    def __init__(self, save_dir='refinement_logs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.feedback_history = []
        self.logger = logging.getLogger(__name__)
        init()  # Initialize colorama
        
    def get_key(self):
        """Get a single keypress without echo"""
        return msvcrt.getch().decode('utf-8').lower()
    
    def get_user_feedback(self, image_number, predicted_level, total_images=12):
        """Get user feedback for a single image with enhanced controls"""
        while True:
            print("\n" + "="*50)
            print(f"{Fore.CYAN}Image {image_number+1}/{total_images}{Style.RESET_ALL}")
            print(f"Model predicted traffic level: {Fore.YELLOW}{predicted_level}{Style.RESET_ALL}")
            print("\nControls:")
            print(f"• {Fore.GREEN}Y/1{Style.RESET_ALL} - Agree with prediction")
            print(f"• {Fore.RED}N/0{Style.RESET_ALL} - Disagree with prediction")
            print(f"• {Fore.BLUE}B{Style.RESET_ALL} - Go back to previous image")
            print(f"• {Fore.YELLOW}Q{Style.RESET_ALL} - Quit session")
            print("\nDo you agree with this classification?")
            
            try:
                key = msvcrt.getch().decode('utf-8').lower()
                
                if key in ['y', '1']:
                    return int(predicted_level), True, 'continue'
                elif key in ['n', '0']:
                    while True:
                        print(f"\n{Fore.YELLOW}Enter correct traffic level (0-4):{Style.RESET_ALL}")
                        print("0: No traffic")
                        print("1: Light traffic")
                        print("2: Moderate traffic")
                        print("3: Heavy traffic")
                        print("4: Severe congestion")
                        print(f"{Fore.BLUE}B - Go back{Style.RESET_ALL}")
                        
                        correction_key = msvcrt.getch().decode('utf-8').lower()
                        
                        if correction_key in ['0', '1', '2', '3', '4']:
                            return int(correction_key), False, 'continue'
                        elif correction_key == 'b':
                            break  # Go back to main menu
                        elif correction_key == 'q':
                            return None, None, 'quit'
                            
                elif key == 'b':
                    return None, None, 'back'
                elif key == 'q':
                    return None, None, 'quit'
                
            except Exception as e:
                self.logger.error(f"Error reading input: {str(e)}")
                print(f"{Fore.RED}Error reading input. Please try again.{Style.RESET_ALL}")
                continue
            
    def save_feedback(self, feedback_data):
        """Save feedback with session metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Add session metadata
        feedback_data['metadata'] = {
            'timestamp': timestamp,
            'total_images': len(feedback_data['feedback']),
            'correct_predictions': sum(1 for f in feedback_data['feedback'] if f['was_correct']),
            'session_duration': feedback_data.get('session_duration', 0)
        }
        
        filename = os.path.join(self.save_dir, f'feedback_{timestamp}.json')
        
        try:
            with open(filename, 'w') as f:
                json.dump(feedback_data, f, indent=4)
            print(f"\n{Fore.GREEN}Feedback saved successfully to {filename}{Style.RESET_ALL}")
        except Exception as e:
            self.logger.error(f"Error saving feedback: {str(e)}")
            print(f"{Fore.RED}Error saving feedback. Check logs for details.{Style.RESET_ALL}") 