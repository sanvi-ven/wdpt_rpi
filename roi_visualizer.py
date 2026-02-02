#!/usr/bin/env python3
"""
ROI Visualizer for Raspberry Pi
Helps you identify and adjust Region of Interest coordinates for WDPT analysis.
Works without GUI (headless) - saves images for inspection.
"""

import cv2
import numpy as np
import os
import time

class ROIVisualizer:
    def __init__(self):
        # Default ROI coordinates from your working file
        self.roi_presets = {
            'current': (684, 185, 984, 377),
            'working_file_1': (713, 248, 1150, 452),
            'working_file_2': (636, 233, 956, 450),
            'center_crop': (500, 200, 800, 500),
        }
        
        # Create output directory
        self.output_dir = "roi_analysis"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def capture_test_frame(self):
        """Capture a single frame for ROI testing"""
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
            
        # Camera warm-up
        print("Warming up camera...")
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                cap.release()
                return None
                
        # Capture the test frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"Captured frame: {frame.shape}")
            return frame
        else:
            print("Error: Could not capture frame")
            return None
            
    def visualize_roi_presets(self, frame):
        """Test all preset ROI coordinates"""
        print("\nTesting ROI presets...")
        
        for name, (x1, y1, x2, y2) in self.roi_presets.items():
            # Check if ROI is within frame bounds
            h, w = frame.shape[:2]
            if x2 > w or y2 > h or x1 < 0 or y1 < 0:
                print(f"Warning: ROI '{name}' {(x1, y1, x2, y2)} exceeds frame bounds {(w, h)}")
                continue
                
            # Draw rectangle on full frame
            frame_with_roi = frame.copy()
            cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add text label
            cv2.putText(frame_with_roi, name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save full frame with ROI
            full_filename = f"{self.output_dir}/{name}_full_frame.jpg"
            cv2.imwrite(full_filename, frame_with_roi)
            
            # Extract and save ROI only
            roi_crop = frame[y1:y2, x1:x2]
            roi_filename = f"{self.output_dir}/{name}_roi_crop.jpg"
            cv2.imwrite(roi_filename, roi_crop)
            
            print(f"  {name}: {(x1, y1, x2, y2)} -> Size: {roi_crop.shape[:2]} -> Saved")
            
    def create_grid_visualization(self, frame):
        """Create a grid overlay to help identify coordinates"""
        h, w = frame.shape[:2]
        grid_frame = frame.copy()
        
        # Draw grid lines every 100 pixels
        grid_spacing = 100
        color = (255, 255, 255)  # White lines
        thickness = 1
        
        # Vertical lines
        for x in range(0, w, grid_spacing):
            cv2.line(grid_frame, (x, 0), (x, h), color, thickness)
            # Add coordinate labels
            cv2.putText(grid_frame, str(x), (x+5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        # Horizontal lines
        for y in range(0, h, grid_spacing):
            cv2.line(grid_frame, (0, y), (w, y), color, thickness)
            # Add coordinate labels
            cv2.putText(grid_frame, str(y), (5, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        grid_filename = f"{self.output_dir}/coordinate_grid.jpg"
        cv2.imwrite(grid_filename, grid_frame)
        print(f"Grid visualization saved: {grid_filename}")
        
    def test_custom_roi(self, frame, roi_coords, name="custom"):
        """Test a custom ROI coordinate"""
        x1, y1, x2, y2 = roi_coords
        h, w = frame.shape[:2]
        
        if x2 > w or y2 > h or x1 < 0 or y1 < 0:
            print(f"Error: ROI {roi_coords} exceeds frame bounds {(w, h)}")
            return False
            
        # Draw rectangle
        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
        cv2.putText(frame_with_roi, name, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save files
        full_filename = f"{self.output_dir}/{name}_test_full.jpg"
        cv2.imwrite(full_filename, frame_with_roi)
        
        roi_crop = frame[y1:y2, x1:x2]
        roi_filename = f"{self.output_dir}/{name}_test_crop.jpg"
        cv2.imwrite(roi_filename, roi_crop)
        
        print(f"Custom ROI {roi_coords} -> Size: {roi_crop.shape[:2]} -> Saved")
        return True
        
    def analyze_frame_properties(self, frame):
        """Analyze frame properties"""
        h, w, c = frame.shape
        print(f"\nFrame Analysis:")
        print(f"  Resolution: {w}x{h}")
        print(f"  Channels: {c}")
        print(f"  Data type: {frame.dtype}")
        print(f"  Size in memory: {frame.nbytes / 1024:.1f} KB")
        
        # Save full frame for reference
        full_filename = f"{self.output_dir}/full_frame_reference.jpg"
        cv2.imwrite(full_filename, frame)
        print(f"  Reference frame saved: {full_filename}")
        
    def run_interactive_analysis(self):
        """Main interactive ROI analysis"""
        print("=== ROI Visualizer for Raspberry Pi ===")
        print("This tool helps you identify the correct ROI coordinates")
        print("All images will be saved to:", self.output_dir)
        
        # Capture test frame
        frame = self.capture_test_frame()
        if frame is None:
            return
            
        # Analyze frame properties
        self.analyze_frame_properties(frame)
        
        # Create coordinate grid
        self.create_grid_visualization(frame)
        
        # Test all preset ROIs
        self.visualize_roi_presets(frame)
        
        print(f"\n=== Results saved to {self.output_dir}/ ===")
        print("\nFiles created:")
        files = sorted(os.listdir(self.output_dir))
        for f in files:
            print(f"  {f}")
            
        print(f"\nTo view results:")
        print(f"1. Check the files in {self.output_dir}/")
        print(f"2. Look at *_roi_crop.jpg files to see what each ROI captures")
        print(f"3. Look at *_full_frame.jpg files to see ROI position on full image")
        print(f"4. Use coordinate_grid.jpg to identify custom coordinates")
        
        # Interactive custom ROI testing
        self.interactive_custom_roi(frame)
        
    def interactive_custom_roi(self, frame):
        """Allow testing of custom ROI coordinates"""
        print(f"\n=== Custom ROI Testing ===")
        print("Enter custom ROI coordinates to test (or press Enter to skip)")
        print("Format: x1,y1,x2,y2 (e.g., 600,200,900,400)")
        
        while True:
            try:
                user_input = input("ROI coordinates (or Enter to finish): ").strip()
                if not user_input:
                    break
                    
                coords = [int(x.strip()) for x in user_input.split(',')]
                if len(coords) != 4:
                    print("Error: Please enter 4 coordinates separated by commas")
                    continue
                    
                x1, y1, x2, y2 = coords
                if x1 >= x2 or y1 >= y2:
                    print("Error: Invalid coordinates (x2>x1, y2>y1 required)")
                    continue
                    
                # Test the custom ROI
                timestamp = int(time.time())
                success = self.test_custom_roi(frame, (x1, y1, x2, y2), f"custom_{timestamp}")
                if success:
                    print("ROI saved successfully!")
                    
            except ValueError:
                print("Error: Please enter valid numbers")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
                
def main():
    """Main function for standalone execution"""
    visualizer = ROIVisualizer()
    visualizer.run_interactive_analysis()
    
if __name__ == "__main__":
    main()