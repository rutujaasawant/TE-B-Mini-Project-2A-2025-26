import asyncio
import json
import logging
import platform
from typing import Dict, Optional
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import av

logger = logging.getLogger(__name__)

class WebRTCHandler:
    """Handles WebRTC connections for voice and video calls"""
    
    def __init__(self):
        self.active_connections: Dict[str, RTCPeerConnection] = {}
        self.media_players: Dict[str, MediaPlayer] = {}
        self.media_recorders: Dict[str, MediaRecorder] = {}
        self.is_windows = platform.system() == "Windows"
        
    async def create_peer_connection(self, user_id: str, room_id: str) -> RTCPeerConnection:
        """Create a new WebRTC peer connection"""
        try:
            pc = RTCPeerConnection()
            
            # Store the connection
            connection_id = f"{room_id}_{user_id}"
            self.active_connections[connection_id] = pc
            
            # Handle connection state changes
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state changed: {pc.connectionState}")
                if pc.connectionState == "failed":
                    await self.cleanup_connection(connection_id)
            
            # Handle ICE connection state changes
            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                logger.info(f"ICE connection state: {pc.iceConnectionState}")
            
            # Handle track events
            @pc.on("track")
            def on_track(track: MediaStreamTrack):
                logger.info(f"Track received: {track.kind}")
                
                if track.kind == "audio":
                    # Handle audio track
                    pass
                elif track.kind == "video":
                    # Handle video track
                    pass
                
                @track.on("ended")
                async def on_ended():
                    logger.info(f"Track ended: {track.kind}")
            
            return pc
            
        except Exception as e:
            logger.error(f"Error creating peer connection: {e}")
            raise
    
    async def handle_offer(self, user_id: str, room_id: str, offer_data: dict) -> dict:
        """Handle incoming WebRTC offer"""
        try:
            pc = await self.create_peer_connection(user_id, room_id)
            
            # Set remote description
            offer = RTCSessionDescription(
                sdp=offer_data["sdp"],
                type=offer_data["type"]
            )
            await pc.setRemoteDescription(offer)
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            await self.cleanup_connection(f"{room_id}_{user_id}")
            raise
    
    async def handle_answer(self, user_id: str, room_id: str, answer_data: dict):
        """Handle incoming WebRTC answer"""
        try:
            connection_id = f"{room_id}_{user_id}"
            pc = self.active_connections.get(connection_id)
            
            if pc:
                answer = RTCSessionDescription(
                    sdp=answer_data["sdp"],
                    type=answer_data["type"]
                )
                await pc.setRemoteDescription(answer)
                
        except Exception as e:
            logger.error(f"Error handling answer: {e}")
    
    async def handle_ice_candidate(self, user_id: str, room_id: str, candidate_data: dict):
        """Handle incoming ICE candidate"""
        try:
            connection_id = f"{room_id}_{user_id}"
            pc = self.active_connections.get(connection_id)
            
            if pc:
                await pc.addIceCandidate(candidate_data)
                
        except Exception as e:
            logger.error(f"Error handling ICE candidate: {e}")
    
    async def start_media_stream(self, user_id: str, room_id: str, media_type: str = "both"):
        """Start local media stream (camera/microphone)"""
        try:
            connection_id = f"{room_id}_{user_id}"
            pc = self.active_connections.get(connection_id)
            
            if not pc:
                return
            
            # Create media player based on platform
            if media_type in ["video", "both"]:
                try:
                    if self.is_windows:
                        # Windows: try different camera sources
                        try:
                            # Try DirectShow camera
                            player = MediaPlayer("video=Webcam", format="dshow", options={"video_size": "640x480"})
                        except:
                            try:
                                # Try default camera
                                player = MediaPlayer("default", format="dshow", options={"video_size": "640x480"})
                            except:
                                # Fallback to any available camera
                                player = MediaPlayer("video=*", format="dshow", options={"video_size": "640x480"})
                    else:
                        # Linux/Mac
                        player = MediaPlayer("/dev/video0", format="v4l2", options={"video_size": "640x480"})
                    
                    self.media_players[connection_id] = player
                    
                    # Add video track
                    if hasattr(player, 'video') and player.video:
                        pc.addTrack(player.video)
                        logger.info("Video track added successfully")
                    else:
                        logger.warning("No video track available")
                        
                except Exception as e:
                    logger.error(f"Error starting video stream: {e}")
                    # Continue with audio only if video fails
            
            if media_type in ["audio", "both"]:
                try:
                    if self.is_windows:
                        # Windows: try different audio sources
                        try:
                            # Try DirectShow microphone
                            player = MediaPlayer("audio=Microphone", format="dshow")
                        except:
                            try:
                                # Try default audio
                                player = MediaPlayer("default", format="dshow")
                            except:
                                # Fallback to any available audio
                                player = MediaPlayer("audio=*", format="dshow")
                    else:
                        # Linux/Mac
                        player = MediaPlayer("default", format="pulse")
                    
                    self.media_players[connection_id] = player
                    
                    # Add audio track
                    if hasattr(player, 'audio') and player.audio:
                        pc.addTrack(player.audio)
                        logger.info("Audio track added successfully")
                    else:
                        logger.warning("No audio track available")
                        
                except Exception as e:
                    logger.error(f"Error starting audio stream: {e}")
                
        except Exception as e:
            logger.error(f"Error starting media stream: {e}")
    
    async def stop_media_stream(self, user_id: str, room_id: str):
        """Stop local media stream"""
        try:
            connection_id = f"{room_id}_{user_id}"
            
            # Stop media player
            if connection_id in self.media_players:
                player = self.media_players[connection_id]
                player.stop()
                del self.media_players[connection_id]
                
        except Exception as e:
            logger.error(f"Error stopping media stream: {e}")
    
    async def start_recording(self, user_id: str, room_id: str, output_path: str):
        """Start recording the call"""
        try:
            connection_id = f"{room_id}_{user_id}"
            pc = self.active_connections.get(connection_id)
            
            if pc:
                recorder = MediaRecorder(output_path)
                self.media_recorders[connection_id] = recorder
                
                # Add recorder to all tracks
                for sender in pc.getSenders():
                    if sender.track:
                        recorder.addTrack(sender.track)
                
                await recorder.start()
                
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
    
    async def stop_recording(self, user_id: str, room_id: str):
        """Stop recording the call"""
        try:
            connection_id = f"{room_id}_{user_id}"
            
            if connection_id in self.media_recorders:
                recorder = self.media_recorders[connection_id]
                await recorder.stop()
                del self.media_recorders[connection_id]
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
    
    async def cleanup_connection(self, connection_id: str):
        """Clean up a WebRTC connection"""
        try:
            # Close peer connection
            if connection_id in self.active_connections:
                pc = self.active_connections[connection_id]
                await pc.close()
                del self.active_connections[connection_id]
            
            # Stop media player
            if connection_id in self.media_players:
                player = self.media_players[connection_id]
                player.stop()
                del self.media_players[connection_id]
            
            # Stop recording
            if connection_id in self.media_recorders:
                recorder = self.media_recorders[connection_id]
                await recorder.stop()
                del self.media_recorders[connection_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    async def get_connection_stats(self, user_id: str, room_id: str) -> dict:
        """Get connection statistics"""
        try:
            connection_id = f"{room_id}_{user_id}"
            pc = self.active_connections.get(connection_id)
            
            if pc:
                stats = await pc.getStats()
                return {
                    "connection_state": pc.connectionState,
                    "ice_connection_state": pc.iceConnectionState,
                    "stats": stats
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}
    
    def get_active_connections_count(self) -> int:
        """Get count of active connections"""
        return len(self.active_connections)
    
    def get_available_devices(self) -> dict:
        """Get available audio/video devices (Windows-specific)"""
        devices = {
            "audio": [],
            "video": []
        }
        
        if self.is_windows:
            try:
                # This would require additional Windows-specific libraries
                # For now, return common device names
                devices["audio"] = ["Microphone", "Default Audio Device"]
                devices["video"] = ["Webcam", "Default Camera"]
            except:
                pass
        
        return devices

# Global WebRTC handler instance
webrtc_handler = WebRTCHandler()
