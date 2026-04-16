import os
import shutil
import csv
from ultralytics import YOLO

def main():
    print("Loading YOLOv8 Nano model...")
    # Load a pre-trained YOLOv8 model
    model = YOLO("yolov8s.pt")
    
    print("Starting training process...")
    # Train the model using the provided dataset configuration
    # 30 epochs is a good starting point for fine-tuning.
    epochs = 50
    batch = 64
    
    results = model.train(
        data="data.yaml", 
        epochs=epochs, 
        imgsz=416, # Roboflow readme mentions 416x416
        patience=10, # Stop early if no improvement for 10 epochs
        batch=batch, # Adjust if you run into memory errors
        project="runs/detect", # Where to save results
        name="train", # Folder name
        exist_ok=True # Overwrite existing training folder if it exists
    )
    
    print("Training complete!")
    save_dir = str(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else "runs/detect/train"
    best_weight_path = os.path.join(save_dir, "weights", "best.pt")
    print(f"Your best fine-tuned model is saved at: {best_weight_path}")
    
    trained_models_dir = r"c:\Users\Work\Desktop\Parking_Detection_Project\trained_models"
    run_name = f"epo{epochs}bat{batch}"
    run_dir = os.path.join(trained_models_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    if os.path.exists(best_weight_path):
        target_name = f"{run_name}.pt"
        target_path = os.path.join(run_dir, target_name)
        shutil.copy(best_weight_path, target_path)
        print(f"Model successfully saved to {target_path}")
        
        # Extract and save training metrics
        results_csv_path = os.path.join(save_dir, "results.csv")
        if os.path.exists(results_csv_path):
            with open(results_csv_path, mode='r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
                if len(reader) > 1:
                    headers = [h.strip() for h in reader[0]]
                    last_row = [v.strip() for v in reader[-1]]
                    
                    txt_name = f"{run_name}.txt"
                    txt_path = os.path.join(run_dir, txt_name)
                    
                    print("\n--- Final Training Metrics ---")
                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        txt_file.write("Training Results (Last Epoch):\n")
                        txt_file.write("-" * 30 + "\n")
                        for k, v in zip(headers, last_row):
                            line = f"{k}: {v}"
                            txt_file.write(line + "\n")
                            # Only print accuracy and loss metrics to console to avoid clutter
                            if any(x in k.lower() for x in ['loss', 'map', 'precision', 'recall']):
                                print(line)
                    
                    print(f"\nMetrics successfully saved to {txt_path}\n")
                    
                    # Generate and save accuracy/loss plot
                    try:
                        import matplotlib.pyplot as plt
                        epochs_list = []
                        train_loss = []
                        val_loss = []
                        acc_map = []
                        
                        def get_col_idx(name_substring):
                            for i, h in enumerate(headers):
                                if name_substring in h: return i
                            return -1
                            
                        epoch_idx = get_col_idx('epoch')
                        t_box_idx = get_col_idx('train/box_loss')
                        t_cls_idx = get_col_idx('train/cls_loss')
                        v_box_idx = get_col_idx('val/box_loss')
                        v_cls_idx = get_col_idx('val/cls_loss')
                        map50_idx = get_col_idx('metrics/mAP50(B)')
                        
                        if all(idx >= 0 for idx in [epoch_idx, t_box_idx, t_cls_idx, v_box_idx, v_cls_idx, map50_idx]):
                            for row in reader[1:]:
                                if len(row) < len(headers): continue
                                epochs_list.append(float(row[epoch_idx]))
                                train_loss.append(float(row[t_box_idx]) + float(row[t_cls_idx]))
                                val_loss.append(float(row[v_box_idx]) + float(row[v_cls_idx]))
                                acc_map.append(float(row[map50_idx]))
                                
                            plt.figure(figsize=(10, 4))
                            
                            # Loss Subplot
                            plt.subplot(1, 2, 1)
                            plt.plot(epochs_list, train_loss, label='Train Loss')
                            plt.plot(epochs_list, val_loss, label='Val Loss')
                            plt.title('Loss (Box + Cls)')
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.legend()
                            
                            # Accuracy Subplot
                            plt.subplot(1, 2, 2)
                            plt.plot(epochs_list, acc_map, label='mAP50', color='green')
                            plt.title('Accuracy (mAP50)')
                            plt.xlabel('Epoch')
                            plt.ylabel('Score')
                            plt.legend()
                            
                            plt.tight_layout()
                            plot_name = f"{run_name}_plot.png"
                            plot_path = os.path.join(run_dir, plot_name)
                            plt.savefig(plot_path)
                            plt.close()
                            print(f"Plot chart successfully saved to {plot_path}")
                        else:
                            print("Plot not generated: could not find expected columns in results.csv.")
                    except ImportError:
                        print("matplotlib is not installed, skipping plot generation. Run 'pip install matplotlib' to enable.")
                    except Exception as e:
                        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    # Prevent multi-processing issues on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
