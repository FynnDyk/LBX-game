import * as THREE from 'three';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { getWeaponById } from './catalog';
import { CustomModelAsset, getCachedCustomModel } from './customModelStorage';
import { buildHangarWeaponGroups, LbxRig } from './lbxModel';
import { createContactShadowTexture } from './textures';
import { LbxLoadout, ModelStyle } from './types';

export interface BoundBone {
  node: THREE.Object3D;
  restLocalQuat: THREE.Quaternion;
  parentInCharQuat: THREE.Quaternion;
}

export interface CustomBoneSet {
  spineLower?: BoundBone;
  spineUpper?: BoundBone;
  neck?: BoundBone;
  head?: BoundBone;
  shoulderL?: BoundBone;
  elbowL?: BoundBone;
  handL?: BoundBone;
  weaponL?: BoundBone;
  shoulderR?: BoundBone;
  elbowR?: BoundBone;
  handR?: BoundBone;
  weaponR?: BoundBone;
  hipL?: BoundBone;
  kneeL?: BoundBone;
  footL?: BoundBone;
  hipR?: BoundBone;
  kneeR?: BoundBone;
  footR?: BoundBone;
  tail?: BoundBone;
  capeL: BoundBone[];
  capeR: BoundBone[];
}

export interface CustomBonePose {
  bodyYaw: number;
  bodyPitch: number;
  bodyRoll: number;
  bodyBob: number;
  torsoYaw: number;
  torsoPitch: number;
  torsoRoll: number;
  headYaw: number;
  headPitch: number;
  lHipX: number;
  lHipZ: number;
  rHipX: number;
  rHipZ: number;
  lKneeX: number;
  rKneeX: number;
  lFootX: number;
  rFootX: number;
  lShoulderX: number;
  lShoulderY: number;
  lShoulderZ: number;
  lElbowX: number;
  lHandX: number;
  lHandY: number;
  lHandZ: number;
  lWeaponX: number;
  rShoulderX: number;
  rShoulderY: number;
  rShoulderZ: number;
  rElbowX: number;
  rHandX: number;
  rHandY: number;
  rHandZ: number;
  rWeaponX: number;
  tailPitch: number;
  tailYaw: number;
  capeLift: number;
}

export interface CustomLbxRig extends LbxRig {
  mixer?: THREE.AnimationMixer;
  specialMixer?: THREE.AnimationMixer;
  specialAction?: THREE.AnimationAction;
  specialClipDuration?: number;
  specialAnimActive?: boolean;
  poseSnapshot?: Array<{
    node: THREE.Object3D;
    pos: THREE.Vector3;
    quat: THREE.Quaternion;
    scale: THREE.Vector3;
  }>;
  backMissiles?: THREE.Object3D[];
  isCustomModel?: boolean;
  customCharacterId?: ModelStyle;
  animationCount?: number;
  hasSkeleton?: boolean;
  isPartArticulated?: boolean;
  hasBuiltInWeapon?: boolean;
  hasHangarWeapon?: boolean;
  bones?: CustomBoneSet;
  poseState?: CustomBonePose;
  animTime?: number;
  attackSlashMesh?: THREE.Mesh;
}

export function startCustomSpecialClip(rig: CustomLbxRig, durationSec: number): boolean {
  if (!rig.specialMixer || !rig.specialAction) return false;
  if (rig.poseSnapshot) {
    for (let i = 0; i < rig.poseSnapshot.length; i += 1) {
      const snap = rig.poseSnapshot[i];
      snap.node.position.copy(snap.pos);
      snap.node.quaternion.copy(snap.quat);
      snap.node.scale.copy(snap.scale);
    }
  }
  const clipDur = Math.max(0.15, rig.specialClipDuration ?? 0.9);
  const scale = THREE.MathUtils.clamp(clipDur / Math.max(0.25, durationSec), 0.08, 3.0);
  rig.specialAction.reset();
  rig.specialAction.setLoop(THREE.LoopOnce, 1);
  rig.specialAction.clampWhenFinished = true;
  rig.specialAction.timeScale = scale;
  rig.specialAction.play();
  rig.specialAnimActive = true;
  return true;
}

export function stopCustomSpecialClip(rig: CustomLbxRig): void {
  if (!rig.specialAnimActive) return;
  rig.specialAction?.stop();
  rig.specialMixer?.setTime(0);
  if (rig.poseSnapshot) {
    for (let i = 0; i < rig.poseSnapshot.length; i += 1) {
      const snap = rig.poseSnapshot[i];
      snap.node.position.copy(snap.pos);
      snap.node.quaternion.copy(snap.quat);
      snap.node.scale.copy(snap.scale);
    }
  }
  rig.specialAnimActive = false;
}

const _euler = new THREE.Euler();
const _qCharDelta = new THREE.Quaternion();
const _qInv = new THREE.Quaternion();
const _qParentDelta = new THREE.Quaternion();

function rotateBoneInCharSpace(
  bone: BoundBone | undefined,
  rx: number,
  ry: number,
  rz: number,
) {
  if (!bone) return;
  if (Math.abs(rx) < 0.0005 && Math.abs(ry) < 0.0005 && Math.abs(rz) < 0.0005) {
    bone.node.quaternion.copy(bone.restLocalQuat);
    return;
  }
  _euler.set(rx, ry, rz, 'YXZ');
  _qCharDelta.setFromEuler(_euler);
  _qInv.copy(bone.parentInCharQuat).invert();
  _qParentDelta
    .copy(_qInv)
    .multiply(_qCharDelta)
    .multiply(bone.parentInCharQuat);
  bone.node.quaternion.copy(_qParentDelta).multiply(bone.restLocalQuat);
}

export function createZeroCustomPose(): CustomBonePose {
  return {
    bodyYaw: 0,
    bodyPitch: 0,
    bodyRoll: 0,
    bodyBob: 0,
    torsoYaw: 0,
    torsoPitch: 0,
    torsoRoll: 0,
    headYaw: 0,
    headPitch: 0,
    lHipX: 0,
    lHipZ: 0,
    rHipX: 0,
    rHipZ: 0,
    lKneeX: 0,
    rKneeX: 0,
    lFootX: 0,
    rFootX: 0,
    lShoulderX: 0,
    lShoulderY: 0,
    lShoulderZ: 0,
    lElbowX: 0,
    lHandX: 0,
    lHandY: 0,
    lHandZ: 0,
    lWeaponX: 0,
    rShoulderX: 0,
    rShoulderY: 0,
    rShoulderZ: 0,
    rElbowX: 0,
    rHandX: 0,
    rHandY: 0,
    rHandZ: 0,
    rWeaponX: 0,
    tailPitch: 0,
    tailYaw: 0,
    capeLift: 0,
  };
}

function easeOutCubic(t: number): number {
  const inv = 1 - THREE.MathUtils.clamp(t, 0, 1);
  return 1 - inv * inv * inv;
}

function easeOutQuart(t: number): number {
  const inv = 1 - THREE.MathUtils.clamp(t, 0, 1);
  return 1 - inv * inv * inv * inv;
}

function easeInOutQuad(t: number): number {
  const x = THREE.MathUtils.clamp(t, 0, 1);
  return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
}

function smoothstep01(t: number): number {
  const x = THREE.MathUtils.clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

function lerpPose(from: CustomBonePose, to: CustomBonePose, t: number): CustomBonePose {
  const out = createZeroCustomPose();
  const keys = Object.keys(out) as (keyof CustomBonePose)[];
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    out[k] = THREE.MathUtils.lerp(from[k], to[k], t);
  }
  return out;
}

function blendFourStageMotion(
  ready: CustomBonePose,
  windup: CustomBonePose,
  strike: CustomBonePose,
  followThrough: CustomBonePose,
  p: number,
  tWindup = 0.25,
  tStrike = 0.50,
  tFollow = 0.68,
): CustomBonePose {
  if (p < tWindup) {
    return lerpPose(ready, windup, smoothstep01(p / tWindup));
  }
  if (p < tStrike) {
    return lerpPose(windup, strike, easeOutQuart((p - tWindup) / (tStrike - tWindup)));
  }
  if (p < tFollow) {
    return lerpPose(strike, followThrough, smoothstep01((p - tStrike) / (tFollow - tStrike)));
  }
  return lerpPose(followThrough, ready, easeInOutQuad((p - tFollow) / (1 - tFollow)));
}

type MotionEase = 'smooth' | 'cubic' | 'quart' | 'quad';

function evalMotionEase(ease: MotionEase | undefined, t: number): number {
  if (ease === 'quart') return easeOutQuart(t);
  if (ease === 'cubic') return easeOutCubic(t);
  if (ease === 'quad') return easeInOutQuad(t);
  return smoothstep01(t);
}

function blendMultiStageMotion(
  keyframes: Array<{ t: number; pose: CustomBonePose; ease?: MotionEase }>,
  p: number,
): CustomBonePose {
  const clamped = THREE.MathUtils.clamp(p, 0, 1);
  for (let i = 0; i < keyframes.length - 1; i++) {
    const cur = keyframes[i];
    const next = keyframes[i + 1];
    if (clamped <= next.t || i === keyframes.length - 2) {
      const span = Math.max(0.0001, next.t - cur.t);
      const localT = THREE.MathUtils.clamp((clamped - cur.t) / span, 0, 1);
      return lerpPose(cur.pose, next.pose, evalMotionEase(next.ease, localT));
    }
  }
  return keyframes[keyframes.length - 1].pose;
}

function applyFluidGait(
  pose: CustomBonePose,
  params: {
    stridePhase: number;
    speedRatio: number;
    backward?: boolean;
    strafeRatio?: number;
    baseYaw?: number;
    leanScale?: number;
  },
) {
  const {
    stridePhase,
    speedRatio,
    backward = false,
    strafeRatio = 0,
    baseYaw = 0,
    leanScale = 1,
  } = params;
  const s = THREE.MathUtils.clamp(speedRatio, 0, 1.45);
  const sprint = THREE.MathUtils.clamp((s - 0.52) / 0.68, 0, 1);
  const dirSign = backward ? -1 : 1;
  const strafe = THREE.MathUtils.clamp(strafeRatio, -1.2, 1.2);
  const absStrafe = Math.min(1, Math.abs(strafe));

  // Biomechanical gait: asymmetric thigh sweep, crisp swing knee-drive,
  // heel-strike cushion, and plantarflexion toe push-off
  const sinP = Math.sin(stridePhase);
  const cosP = Math.cos(stridePhase);
  const swingAmp = (0.5 + 0.28 * sprint) * (1 - absStrafe * 0.22);

  const swingFlexL = Math.pow(Math.max(0, Math.sin(stridePhase * dirSign - 0.62)), 0.76);
  const swingFlexR = Math.pow(Math.max(0, -Math.sin(stridePhase * dirSign - 0.62)), 0.76);
  const plantCushionL = Math.pow(Math.max(0, -sinP * dirSign), 1.12);
  const plantCushionR = Math.pow(Math.max(0, sinP * dirSign), 1.12);
  const pushOffL = Math.max(0, cosP * dirSign);
  const pushOffR = Math.max(0, -cosP * dirSign);

  pose.lHipX = (sinP * swingAmp + Math.sin(stridePhase * 2) * 0.045) * s;
  pose.rHipX = (-sinP * swingAmp + Math.sin(stridePhase * 2) * 0.045) * s;
  pose.lKneeX = (swingFlexL * (0.82 + 0.38 * sprint) + plantCushionL * 0.3) * s + 0.14;
  pose.rKneeX = (swingFlexR * (0.82 + 0.38 * sprint) + plantCushionR * 0.3) * s + 0.14;

  // Natural ankle roll: toe push-off extension + dorsiflexion clearance in swing
  pose.lFootX =
    -(pose.bodyPitch + pose.lHipX + pose.lKneeX) * 0.44 +
    pushOffL * 0.28 * s -
    swingFlexL * 0.16 * s;
  pose.rFootX =
    -(pose.bodyPitch + pose.rHipX + pose.rKneeX) * 0.44 +
    pushOffR * 0.28 * s -
    swingFlexR * 0.16 * s;

  // Wider athletic stance and outward foot cut when strafing laterally
  pose.lHipZ = 0.06 + Math.max(0, -strafe) * 0.13 + absStrafe * 0.03;
  pose.rHipZ = -0.06 - Math.max(0, strafe) * 0.13 - absStrafe * 0.03;

  // Double-frequency vertical center-of-mass drop on heel strike + hip sway & counter-twist
  pose.bodyBob = (-Math.cos(stridePhase * 2 + 0.35) * (0.052 + 0.014 * sprint) + 0.01) * s;
  pose.bodyYaw = baseYaw - cosP * (0.09 + 0.055 * sprint) * s - strafe * 0.07;
  pose.bodyPitch =
    (backward ? -0.11 : 0.13 + 0.16 * sprint) * Math.min(1, s) * leanScale +
    Math.sin(stridePhase * 2) * 0.028 * s;
  pose.bodyRoll = strafe * (0.15 + 0.07 * sprint) - sinP * (0.045 + 0.02 * sprint) * s;

  // Counter-rotate upper torso & shoulders while head stabilizes on the horizon
  const counterTwist = cosP * (0.14 + 0.08 * sprint) * s;
  pose.torsoYaw += counterTwist + strafe * 0.09;
  pose.torsoRoll += sinP * 0.036 * s - strafe * 0.04;
  pose.headYaw -= counterTwist * 0.72 + strafe * 0.05;
  pose.headPitch -= Math.sin(stridePhase * 2) * 0.022 * s;
  pose.tailPitch = (0.16 + 0.2 * sprint) * Math.min(1, s) + Math.abs(sinP) * 0.16 * s;
  pose.tailYaw = -sinP * (0.32 + 0.18 * sprint) * s - strafe * 0.34;
  pose.capeLift = (0.34 + 0.4 * sprint) * s + Math.abs(Math.sin(stridePhase * 2)) * 0.16 * s;
}

function applyFluidAirborneLegs(
  pose: CustomBonePose,
  velY: number,
  elapsed = 0,
  strafeRatio = 0,
) {
  const rise = THREE.MathUtils.clamp(velY / 7.5, -1, 1);
  const strafe = THREE.MathUtils.clamp(strafeRatio, -1, 1);
  const hoverBreath = Math.sin(elapsed * 5.2) * 0.024;

  if (rise > 0.56) {
    // Explosive launch push-off transitioning into athletic lead-knee drive
    const launch = THREE.MathUtils.clamp((rise - 0.56) / 0.44, 0, 1);
    pose.bodyPitch = THREE.MathUtils.lerp(-0.15, -0.22, launch);
    pose.lHipX = THREE.MathUtils.lerp(-0.96, -1.04, launch);
    pose.lKneeX = THREE.MathUtils.lerp(1.36, 1.44, launch);
    pose.rHipX = THREE.MathUtils.lerp(0.42, 0.58, launch);
    pose.rKneeX = THREE.MathUtils.lerp(0.32, 0.18, launch);
    pose.lFootX = 0.44;
    pose.rFootX = 0.56;
    pose.capeLift = THREE.MathUtils.lerp(0.74, 0.54, launch);
  } else if (rise > 0.18) {
    // Mid-ascent tucked knee-drive
    const a = THREE.MathUtils.clamp((rise - 0.18) / 0.38, 0, 1);
    pose.bodyPitch = THREE.MathUtils.lerp(0.02, -0.15, a);
    pose.lHipX = THREE.MathUtils.lerp(-0.7, -0.96, a);
    pose.lKneeX = THREE.MathUtils.lerp(1.08, 1.36, a);
    pose.rHipX = THREE.MathUtils.lerp(0.28, 0.42, a);
    pose.rKneeX = THREE.MathUtils.lerp(0.64, 0.32, a);
    pose.lFootX = 0.24 + 0.2 * a;
    pose.rFootX = 0.28 + 0.24 * a;
    pose.capeLift = THREE.MathUtils.lerp(0.96, 0.74, a);
  } else if (rise < -0.2) {
    // Descent landing prep: legs reach cleanly down, ankles flex to absorb impact
    const f = THREE.MathUtils.clamp((-rise - 0.2) / 0.8, 0, 1);
    pose.bodyPitch = THREE.MathUtils.lerp(0.02, 0.19, f);
    pose.lHipX = THREE.MathUtils.lerp(-0.7, -0.32, f);
    pose.lKneeX = THREE.MathUtils.lerp(1.08, 0.52, f);
    pose.rHipX = THREE.MathUtils.lerp(0.28, 0.14, f);
    pose.rKneeX = THREE.MathUtils.lerp(0.64, 0.44, f);
    pose.lFootX = THREE.MathUtils.lerp(0.24, -0.2, f);
    pose.rFootX = THREE.MathUtils.lerp(0.28, -0.16, f);
    pose.capeLift = THREE.MathUtils.lerp(0.96, 1.32, f);
  } else {
    // Weightless apex hang-time with subtle micro-stabilization
    pose.bodyPitch = 0.02 + hoverBreath;
    pose.lHipX = -0.7 + hoverBreath * 0.4;
    pose.lKneeX = 1.08;
    pose.rHipX = 0.28 - hoverBreath * 0.4;
    pose.rKneeX = 0.64;
    pose.lFootX = 0.24;
    pose.rFootX = 0.28;
    pose.capeLift = 0.96 + hoverBreath * 1.8;
  }
  pose.lHipZ = 0.08 + Math.max(0, -strafe) * 0.08;
  pose.rHipZ = -0.08 - Math.max(0, strafe) * 0.08;
  pose.bodyBob = 0.03 + hoverBreath * 0.6;
  pose.bodyRoll = strafe * 0.16;
  pose.torsoRoll -= strafe * 0.05;
  pose.tailPitch = rise > 0.18 ? -0.26 : rise < -0.2 ? 0.36 : 0.18 + hoverBreath * 2;
  pose.tailYaw = -strafe * 0.36 + Math.sin(elapsed * 4.5) * 0.09;
}

function applyFluidDashKinematics(
  pose: CustomBonePose,
  params: {
    dashProgress?: number;
    dashForward?: number;
    dashRight?: number;
  },
): { isLateral: boolean; side: number; strength: number } {
  const p = THREE.MathUtils.clamp(params.dashProgress ?? 0.5, 0, 1);
  const dFwd = params.dashForward ?? 1;
  const dRight = params.dashRight ?? 0;

  // Razor-sharp launch burst -> low aerodynamic glide -> crisp braking flare
  const snap = easeOutQuart(Math.min(1, p * 3.0));
  const brake = p > 0.7 ? easeInOutQuad((p - 0.7) / 0.3) : 0;
  const strength = snap * (1 - brake * 0.45);
  const isLateral = Math.abs(dRight) > Math.abs(dFwd) * 0.9 && Math.abs(dRight) > 0.35;
  const side = dRight >= 0 ? 1 : -1;

  if (isLateral) {
    // Dedicated Left / Right Lateral Jet-Skate Bank
    pose.bodyPitch = (0.16 + brake * 0.08) * strength;
    pose.bodyRoll = side * 0.38 * strength;
    pose.bodyYaw = -side * 0.18 * strength;
    pose.bodyBob = (0.06 - brake * 0.08) * strength;
    pose.torsoYaw += side * 0.28 * strength;
    pose.torsoRoll -= side * 0.07 * strength;
    pose.headYaw -= side * 0.2 * strength;
    if (side > 0) {
      // Dashing RIGHT: right knee tucked aggressively, left leg trails extended to push off
      pose.rHipX = -0.64 * strength;
      pose.rKneeX = 1.08 * strength;
      pose.rFootX = 0.28 * strength;
      pose.lHipX = 0.34 * strength;
      pose.lKneeX = 0.24 * strength;
      pose.lFootX = 0.42 * strength;
      pose.lHipZ = 0.2 * strength;
      pose.rHipZ = -0.08;
    } else {
      // Dashing LEFT: left knee tucked aggressively, right leg trails extended to push off
      pose.lHipX = -0.64 * strength;
      pose.lKneeX = 1.08 * strength;
      pose.lFootX = 0.28 * strength;
      pose.rHipX = 0.34 * strength;
      pose.rKneeX = 0.24 * strength;
      pose.rFootX = 0.42 * strength;
      pose.rHipZ = -0.2 * strength;
      pose.lHipZ = 0.08;
    }
    pose.tailPitch = -0.2 * strength;
    pose.tailYaw = -side * 0.52 * strength;
    pose.capeLift = 1.14 * strength;
  } else if (dFwd < -0.3) {
    // Backward defensive evade glide
    pose.bodyPitch = -0.26 * strength;
    pose.bodyRoll = dRight * 0.18 * strength;
    pose.bodyBob = 0.07 * strength;
    pose.lHipX = -0.46 * strength;
    pose.lKneeX = 0.58 * strength;
    pose.lFootX = -0.14 * strength;
    pose.rHipX = 0.24 * strength;
    pose.rKneeX = 0.82 * strength;
    pose.rFootX = 0.28 * strength;
    pose.lHipZ = 0.07;
    pose.rHipZ = -0.07;
    pose.tailPitch = 0.3 * strength;
    pose.tailYaw = -dRight * 0.28 * strength;
    pose.capeLift = 0.78 * strength;
  } else {
    // Forward aerodynamic jet-boost dive
    pose.bodyPitch = 0.48 * strength;
    pose.bodyRoll = dRight * 0.18 * strength;
    pose.bodyBob = 0.11 * strength;
    pose.lHipX = 0.6 * strength;
    pose.lKneeX = 0.42 * strength;
    pose.lFootX = 0.36 * strength;
    pose.rHipX = 0.2 * strength;
    pose.rKneeX = 0.94 * strength;
    pose.rFootX = 0.4 * strength;
    pose.lHipZ = 0.05;
    pose.rHipZ = -0.05;
    pose.tailPitch = -0.32 * strength;
    pose.tailYaw = -dRight * 0.38 * strength;
    pose.capeLift = 1.18 * strength;
  }

  return { isLateral, side, strength };
}

export function buildCustomRiflePose(params: {
  mode: 'aim' | 'low_ready' | 'fire' | 'dash' | 'air_aim' | 'air_fire';
  elapsed: number;
  speedRatio?: number;
  stridePhase?: number;
  fireProgress?: number;
  shotVariant?: number;
  velY?: number;
  backward?: boolean;
  strafeRatio?: number;
  dashProgress?: number;
  dashForward?: number;
  dashRight?: number;
  isPartRig?: boolean;
  isAiming?: boolean;
}): CustomBonePose {
  const {
    mode,
    elapsed,
    speedRatio = 0,
    stridePhase = 0,
    fireProgress = 0,
    shotVariant = 1,
    velY = 0,
    backward = false,
    strafeRatio = 0,
    dashProgress = 0.5,
    dashForward = 1,
    dashRight = 0,
    isPartRig = false,
    isAiming = false,
  } = params;
  const pose = createZeroCustomPose();
  // Organic multi-frequency breathing + slow weight transfer + predator vigilance scan
  const aimSteady = isAiming ? 0.18 : 1;
  const breath = (Math.sin(elapsed * 1.9) + 0.24 * Math.sin(elapsed * 3.8 + 0.4)) * aimSteady;
  const weightShift = Math.sin(elapsed * 0.85) * aimSteady;
  const scan = isAiming ? 0 : Math.sin(elapsed * 0.68) * 0.055 + Math.sin(elapsed * 1.45) * 0.02;
  const tailWave = Math.sin(elapsed * 1.45) * 0.16 + Math.sin(elapsed * 2.9 + 0.7) * 0.055;

  if (isPartRig) {
    // Dedicated IK-verified Hunter Wolf-Sniper kinematics with living weight-shift & recoil physics
    pose.bodyYaw = (isAiming ? 0.06 : 0.08) + weightShift * 0.032;
    pose.bodyPitch = (isAiming ? 0.07 : 0.04) + breath * 0.012;
    pose.bodyRoll = weightShift * 0.026;
    pose.bodyBob = (isAiming ? -0.07 : -0.045) + breath * 0.013 - Math.abs(weightShift) * 0.006;
    pose.torsoYaw = -0.44 + scan * 0.42;
    pose.torsoPitch = (isAiming ? 0.09 : 0.06) + breath * 0.024;
    pose.torsoRoll = -weightShift * 0.024;
    pose.headYaw = 0.44 + scan * 0.65;
    pose.headPitch = (isAiming ? 0.08 : 0.03) - breath * 0.014;

    // Alert bladed wolf marksman stance with subtle knee weight transfer
    pose.lHipX = -0.24 - weightShift * 0.025;
    pose.lHipZ = 0.07;
    pose.lKneeX = (isAiming ? 0.44 : 0.36) + weightShift * 0.045;
    pose.lFootX = -0.14 - weightShift * 0.02;
    pose.rHipX = 0.22 + weightShift * 0.025;
    pose.rHipZ = -0.07;
    pose.rKneeX = (isAiming ? 0.32 : 0.24) - weightShift * 0.045;
    pose.rFootX = -0.16 + weightShift * 0.02;

    // Right trigger arm + wrist leveling Sniper Rifle straight +Z with subtle breathing micro-poise
    const rifleBreath = breath * 0.022;
    pose.rShoulderX = -1.07 + rifleBreath;
    pose.rShoulderY = 0.72 + scan * 0.25;
    pose.rShoulderZ = 0.05 + weightShift * 0.018;
    pose.rElbowX = -0.49 - breath * 0.015;
    pose.rHandX = 1.155 - rifleBreath * 0.35;
    pose.rHandY = -1.283;
    pose.rHandZ = -1.343;

    // Left support arm cradling rifle forward shroud in sync with breathing
    pose.lShoulderX = -1.26 + rifleBreath;
    pose.lShoulderY = -0.36 + scan * 0.25;
    pose.lShoulderZ = -0.1 + weightShift * 0.015;
    pose.lElbowX = -0.1 - breath * 0.014;
    pose.lHandX = 0.28;
    pose.lHandY = -0.42;
    pose.lHandZ = 0.2;

    pose.tailPitch = 0.17 + breath * 0.045;
    pose.tailYaw = tailWave;
    pose.capeLift = 0.06 + breath * 0.025;

    if (mode === 'dash') {
      pose.torsoYaw = -0.34;
      pose.torsoPitch = 0.14;
      pose.headYaw = 0.32;
      pose.rShoulderX = -0.78;
      pose.rShoulderY = 0.56;
      pose.rShoulderZ = 0.08;
      pose.rElbowX = -0.62;
      pose.rHandX = 1.24;
      pose.rHandY = -0.95;
      pose.rHandZ = -1.05;
      pose.lShoulderX = -0.96;
      pose.lShoulderY = -0.32;
      pose.lShoulderZ = -0.18;
      pose.lElbowX = -0.35;
      const d = applyFluidDashKinematics(pose, { dashProgress, dashForward, dashRight });
      if (d.isLateral) {
        if (d.side > 0) {
          // Dashing RIGHT: keep sniper leveled left-to-forward across chest while banking right
          pose.rShoulderX = -0.94;
          pose.rShoulderY = 0.48 - 0.18 * d.strength;
          pose.lShoulderX = -1.08;
          pose.lShoulderY = -0.22 - 0.14 * d.strength;
          pose.headYaw = 0.24 - 0.18 * d.strength;
        } else {
          // Dashing LEFT: tuck rifle tightly into left shoulder pocket, stabilize muzzle
          pose.rShoulderX = -0.92;
          pose.rShoulderY = 0.68 + 0.14 * d.strength;
          pose.lShoulderX = -1.16;
          pose.lShoulderY = -0.44;
          pose.lShoulderZ = -0.04;
          pose.headYaw = 0.42 + 0.16 * d.strength;
        }
      }
      return pose;
    }

    if (mode === 'air_aim') {
      const rise = THREE.MathUtils.clamp(velY / 7.5, -1, 1);
      pose.bodyYaw = 0.08;
      pose.torsoYaw = -0.44;
      pose.torsoPitch = 0.08 + (rise < -0.2 ? 0.06 : 0);
      pose.headYaw = 0.44;
      pose.headPitch = rise < -0.2 ? 0.12 : 0.05;
      pose.rShoulderX = -1.07 - Math.max(0, rise) * 0.08;
      pose.lShoulderX = -1.26 - Math.max(0, rise) * 0.08;
      applyFluidAirborneLegs(pose, velY, elapsed, strafeRatio);
      return pose;
    }

    if (mode === 'air_fire') {
      const p = THREE.MathUtils.clamp(fireProgress, 0, 1);
      const kick =
        p >= 0.14 && p <= 0.48
          ? Math.sin(((p - 0.14) / 0.34) * Math.PI) * (shotVariant === 3 ? 1.35 : 1.0)
          : 0;
      const settle =
        p > 0.48 && p <= 0.85 ? Math.sin(((p - 0.48) / 0.37) * Math.PI) * 0.18 : 0;
      pose.bodyYaw = 0.1;
      pose.bodyPitch = -0.05 - kick * 0.14 + settle * 0.08;
      pose.bodyBob = 0.04;
      pose.torsoYaw = -0.46 - kick * 0.1;
      pose.torsoPitch = 0.06 - kick * 0.2 + settle * 0.1;
      pose.headYaw = 0.46 + kick * 0.06;
      pose.headPitch = 0.05 - kick * 0.1 + settle * 0.05;

      pose.lHipX = -0.66 + kick * 0.2;
      pose.lKneeX = 1.04 - kick * 0.18;
      pose.rHipX = 0.34 + kick * 0.16;
      pose.rKneeX = 0.48;

      pose.rShoulderX = -1.07 - kick * 0.28 + settle * 0.09;
      pose.rElbowX = -0.49 - kick * 0.11;
      pose.rHandX = 1.155 - kick * 0.08;
      pose.lShoulderX = -1.26 - kick * 0.28 + settle * 0.09;
      pose.lElbowX = -0.1 - kick * 0.09;
      pose.tailPitch = 0.24 + kick * 0.42;
      pose.tailYaw = Math.sin(elapsed * 4.2) * 0.12;
      return pose;
    }

    if (mode === 'low_ready') {
      const s = Math.min(1.4, speedRatio);
      const sway = Math.sin(stridePhase) * 0.07 * s;
      const strideBobArm = Math.cos(stridePhase * 2) * 0.055 * s;
      const shoulderSwing = Math.cos(stridePhase) * 0.06 * s;
      applyFluidGait(pose, {
        stridePhase,
        speedRatio: s,
        backward,
        strafeRatio,
        baseYaw: 0.04,
        leanScale: 0.92,
      });
      // Athletic tactical run with rifle tucked low-ready & natural step bounce
      pose.torsoYaw = -0.38 + sway;
      pose.torsoPitch = 0.13 + strideBobArm * 0.35;
      pose.headYaw = 0.38 - sway * 0.88;
      pose.headPitch = -0.05;
      pose.rShoulderX = -0.8 + strideBobArm + shoulderSwing;
      pose.rShoulderY = 0.62 + sway * 0.4;
      pose.rShoulderZ = 0.08;
      pose.rElbowX = -0.62 - strideBobArm * 0.5;
      pose.rHandX = 1.24 - strideBobArm * 0.4;
      pose.rHandY = -1.05;
      pose.rHandZ = -1.14;
      pose.lShoulderX = -1.0 + strideBobArm + shoulderSwing * 0.85;
      pose.lShoulderY = -0.32 + sway * 0.4;
      pose.lShoulderZ = -0.16;
      pose.lElbowX = -0.3 - strideBobArm * 0.45;
      return pose;
    }

    if (mode === 'fire') {
      const p = THREE.MathUtils.clamp(fireProgress, 0, 1);
      // 4-phase realistic recoil: trigger squeeze brace -> sharp muzzle blast kick -> barrel recovery dip -> settle
      const brace = p < 0.18 ? smoothstep01(p / 0.18) : Math.max(0, 1 - (p - 0.18) / 0.25);

      let kick = 0;
      let recoveryDip = 0;
      if (shotVariant === 2) {
        if (p >= 0.16 && p <= 0.38) {
          kick = Math.sin(((p - 0.16) / 0.22) * Math.PI) * 0.9;
        } else if (p >= 0.46 && p <= 0.7) {
          kick = Math.sin(((p - 0.46) / 0.24) * Math.PI) * 1.02;
        } else if (p > 0.7 && p < 0.95) {
          recoveryDip = Math.sin(((p - 0.7) / 0.25) * Math.PI) * 0.22;
        }
      } else {
        const power = shotVariant === 3 ? 1.42 : 1.04;
        if (p >= 0.16 && p <= 0.48) {
          const kp = (p - 0.16) / 0.32;
          kick = Math.pow(Math.sin(kp * Math.PI), 0.75) * power;
        } else if (p > 0.48 && p < 0.88) {
          recoveryDip = Math.sin(((p - 0.48) / 0.4) * Math.PI) * (shotVariant === 3 ? 0.34 : 0.24);
        }
      }

      pose.bodyBob = -0.055 - brace * 0.032 - kick * 0.035 + recoveryDip * 0.025;
      pose.bodyPitch = 0.05 + brace * 0.03 - kick * 0.11 + recoveryDip * 0.05;
      pose.bodyRoll = kick * 0.03;
      pose.torsoYaw = -0.45 - kick * 0.11 + recoveryDip * 0.04;
      pose.torsoPitch = 0.07 + brace * 0.04 - kick * 0.2 + recoveryDip * 0.08;
      pose.headYaw = 0.45 + kick * 0.07;
      pose.headPitch = 0.04 - kick * 0.09 + recoveryDip * 0.04;

      pose.rShoulderX = -1.07 + brace * 0.03 - kick * 0.29 + recoveryDip * 0.1;
      pose.rElbowX = -0.49 - brace * 0.03 - kick * 0.11 + recoveryDip * 0.04;
      pose.rHandX = 1.155 - kick * 0.08 + recoveryDip * 0.03;
      pose.lShoulderX = -1.26 + brace * 0.03 - kick * 0.28 + recoveryDip * 0.1;
      pose.lElbowX = -0.1 - kick * 0.09;

      pose.lKneeX = 0.38 + brace * 0.08 + kick * 0.06;
      pose.rKneeX = 0.26 + brace * 0.05 + kick * 0.14;
      pose.tailPitch = 0.2 + kick * 0.44 - recoveryDip * 0.12;
      pose.tailYaw = -kick * 0.24 + Math.sin(p * 12) * 0.06 * kick;
      return pose;
    }

    return pose;
  }

  // Standard skeleton Hunter two-handed sniper pose
  pose.bodyYaw = 0.14 + weightShift * 0.03;
  pose.bodyPitch = 0.04 + breath * 0.012;
  pose.bodyRoll = weightShift * 0.024;
  pose.bodyBob = -0.05 + breath * 0.012;
  pose.torsoYaw = -0.6 + scan * 0.35;
  pose.torsoPitch = 0.08 + breath * 0.02;
  pose.torsoRoll = -weightShift * 0.02;
  pose.headYaw = 0.52 + scan * 0.6;
  pose.headPitch = 0.14 - breath * 0.012;

  pose.lHipX = -0.22 - weightShift * 0.02;
  pose.lHipZ = 0.08;
  pose.lKneeX = 0.34 + weightShift * 0.04;
  pose.rHipX = 0.2 + weightShift * 0.02;
  pose.rHipZ = -0.09;
  pose.rKneeX = 0.24 - weightShift * 0.04;

  pose.rShoulderX = -0.36 + breath * 0.02;
  pose.rShoulderY = 0.91 + scan * 0.2;
  pose.rShoulderZ = 0.56;
  pose.rElbowX = -1.57 - breath * 0.015;
  pose.rHandX = 0.4 - breath * 0.01;
  pose.rHandY = 0.55;
  pose.rHandZ = -0.58;

  pose.lShoulderX = -1.15 + breath * 0.02;
  pose.lShoulderY = -0.5 + scan * 0.2;
  pose.lShoulderZ = -0.25;
  pose.lElbowX = -0.45;
  pose.lHandX = 0.02;
  pose.lHandY = 0.25;
  pose.lHandZ = 0.43;
  pose.capeLift = 0.06 + breath * 0.025;

  if (mode === 'dash') {
    pose.torsoPitch = 0.14;
    pose.headPitch = 0.04;
    pose.rShoulderX = -0.22;
    pose.lShoulderX = -1.0;
    pose.rHandX = 0.3;
    const d = applyFluidDashKinematics(pose, { dashProgress, dashForward, dashRight });
    if (d.isLateral) {
      if (d.side > 0) {
        pose.rShoulderX = -0.34;
        pose.rShoulderY = 0.68;
        pose.lShoulderX = -1.12;
      } else {
        pose.rShoulderX = -0.26;
        pose.rShoulderY = 0.96;
        pose.lShoulderX = -1.18;
      }
    }
    return pose;
  }

  if (mode === 'air_aim') {
    const rise = THREE.MathUtils.clamp(velY / 7.5, -1, 1);
    pose.bodyYaw = 0.16;
    pose.torsoYaw = -0.62;
    pose.torsoPitch = 0.1 + (rise < -0.2 ? 0.06 : 0);
    pose.headYaw = 0.54;
    pose.headPitch = 0.12 + (rise < -0.2 ? 0.05 : 0);
    applyFluidAirborneLegs(pose, velY, elapsed, strafeRatio);
    return pose;
  }

  if (mode === 'air_fire') {
    const p = THREE.MathUtils.clamp(fireProgress, 0, 1);
    const kick =
      p >= 0.15 && p <= 0.5
        ? Math.sin(((p - 0.15) / 0.35) * Math.PI) * (shotVariant === 3 ? 1.38 : 1.06)
        : 0;
    const settle = p > 0.5 && p <= 0.85 ? Math.sin(((p - 0.5) / 0.35) * Math.PI) * 0.18 : 0;
    pose.bodyYaw = 0.18;
    pose.bodyPitch = -0.06 - kick * 0.14 + settle * 0.06;
    pose.bodyBob = 0.04;
    pose.torsoYaw = -0.64 - kick * 0.09;
    pose.torsoPitch = 0.1 - kick * 0.2 + settle * 0.08;
    pose.headYaw = 0.56 + kick * 0.06;
    pose.headPitch = 0.15 - kick * 0.1;

    pose.lHipX = -0.66 + kick * 0.18;
    pose.lKneeX = 1.02 - kick * 0.14;
    pose.rHipX = 0.36 + kick * 0.14;
    pose.rKneeX = 0.48;

    pose.rShoulderX = -0.38 - kick * 0.28 + settle * 0.09;
    pose.rElbowX = -1.58 - kick * 0.09;
    pose.rHandX = 0.4 - kick * 0.07;
    pose.lShoulderX = -1.17 - kick * 0.28 + settle * 0.09;
    pose.lElbowX = -0.46 - kick * 0.08;
    pose.capeLift = 0.96 + kick * 0.35;
    return pose;
  }

  if (mode === 'low_ready') {
    const s = Math.min(1.4, speedRatio);
    const sway = Math.sin(stridePhase) * 0.06 * s;
    const armBob = Math.cos(stridePhase * 2) * 0.055 * s;
    pose.torsoYaw = -0.56;
    pose.torsoPitch = 0.11;
    pose.headYaw = 0.5;
    pose.headPitch = 0.08;
    applyFluidGait(pose, {
      stridePhase,
      speedRatio: s,
      backward,
      strafeRatio,
      baseYaw: 0.06,
      leanScale: 0.85,
    });
    pose.torsoYaw = -0.56 + sway;
    pose.headYaw = 0.5 - sway * 0.82;
    pose.rShoulderX = -0.22 + armBob + Math.cos(stridePhase) * 0.05 * s;
    pose.lShoulderX = -1.01 + armBob + Math.cos(stridePhase) * 0.04 * s;
    pose.rHandX = 0.28;
    return pose;
  }

  if (mode === 'fire') {
    const p = THREE.MathUtils.clamp(fireProgress, 0, 1);
    const brace = p < 0.18 ? smoothstep01(p / 0.18) : Math.max(0, 1 - (p - 0.18) / 0.25);

    let kick = 0;
    let recoveryDip = 0;
    if (shotVariant === 2) {
      if (p >= 0.16 && p <= 0.38) {
        kick = Math.sin(((p - 0.16) / 0.22) * Math.PI) * 0.88;
      } else if (p >= 0.46 && p <= 0.7) {
        kick = Math.sin(((p - 0.46) / 0.24) * Math.PI) * 0.98;
      } else if (p > 0.7 && p < 0.95) {
        recoveryDip = Math.sin(((p - 0.7) / 0.25) * Math.PI) * 0.2;
      }
    } else {
      const power = shotVariant === 3 ? 1.4 : 1.04;
      if (p >= 0.16 && p <= 0.48) {
        kick = Math.pow(Math.sin(((p - 0.16) / 0.32) * Math.PI), 0.75) * power;
      } else if (p > 0.48 && p < 0.88) {
        recoveryDip = Math.sin(((p - 0.48) / 0.4) * Math.PI) * 0.24;
      }
    }

    pose.bodyBob = -0.065 - brace * 0.025 - kick * 0.025 + recoveryDip * 0.02;
    pose.bodyPitch = 0.06 + brace * 0.025 - kick * 0.09 + recoveryDip * 0.04;
    pose.torsoYaw = -0.62 - kick * 0.09 + recoveryDip * 0.03;
    pose.torsoPitch = 0.1 + brace * 0.035 - kick * 0.18 + recoveryDip * 0.07;
    pose.headYaw = 0.54 + kick * 0.05;
    pose.headPitch = 0.16 - kick * 0.1 + recoveryDip * 0.04;

    pose.rShoulderX = -0.38 + brace * 0.025 - kick * 0.28 + recoveryDip * 0.09;
    pose.rElbowX = -1.58 - kick * 0.1 + recoveryDip * 0.04;
    pose.rHandX = 0.4 - kick * 0.07;

    pose.lShoulderX = -1.17 + brace * 0.025 - kick * 0.28 + recoveryDip * 0.09;
    pose.lElbowX = -0.46 - kick * 0.08;

    pose.lKneeX = 0.38 + brace * 0.06;
    pose.rKneeX = 0.28 + kick * 0.14;
    pose.capeLift = 0.12 + kick * 0.55;
    return pose;
  }

  return pose;
}

export function buildCustomLancePose(params: {
  mode: 'ready' | 'guard' | 'strike' | 'run' | 'dash' | 'jump' | 'air_strike';
  elapsed?: number;
  speedRatio?: number;
  stridePhase?: number;
  strikeProgress?: number;
  strikeVariant?: number;
  velY?: number;
  backward?: boolean;
  strafeRatio?: number;
  dashProgress?: number;
  dashForward?: number;
  dashRight?: number;
  isSpecial?: boolean;
}): CustomBonePose {
  const {
    mode,
    elapsed = 0,
    speedRatio = 0,
    stridePhase = 0,
    strikeProgress = 0,
    strikeVariant = 1,
    velY = 0,
    backward = false,
    strafeRatio = 0,
    dashProgress = 0.5,
    dashForward = 1,
    dashRight = 0,
    isSpecial = false,
  } = params;
  const breath = Math.sin(elapsed * 2.05) + 0.22 * Math.sin(elapsed * 4.1 + 0.5);
  const weightShift = Math.sin(elapsed * 0.92);
  const poise = Math.sin(elapsed * 1.35 + 0.8);

  // Ready combat hoplite stance with living breathing, hip weight shift, spear & shield poise
  const ready = createZeroCustomPose();
  ready.bodyYaw = -0.08 + weightShift * 0.038;
  ready.bodyPitch = 0.04 + breath * 0.012;
  ready.bodyRoll = weightShift * 0.028;
  ready.bodyBob = -0.04 + breath * 0.015 - Math.abs(weightShift) * 0.008;
  ready.torsoYaw = 0.14 - weightShift * 0.045;
  ready.torsoPitch = 0.04 + breath * 0.024;
  ready.torsoRoll = -weightShift * 0.025;
  ready.headYaw = -0.14 + Math.sin(elapsed * 0.72) * 0.055;
  ready.headPitch = 0.02 - breath * 0.014;

  ready.lHipX = -0.16 - weightShift * 0.028;
  ready.lHipZ = 0.06;
  ready.lKneeX = 0.26 + weightShift * 0.05;
  ready.lFootX = -weightShift * 0.025;
  ready.rHipX = 0.16 + weightShift * 0.028;
  ready.rHipZ = -0.06;
  ready.rKneeX = 0.22 - weightShift * 0.05;
  ready.rFootX = weightShift * 0.025;

  // Shield held dynamically across left flank
  ready.lShoulderX = -0.44 + breath * 0.028 + poise * 0.018;
  ready.lShoulderY = -0.62 + weightShift * 0.03;
  ready.lElbowX = -0.72 - breath * 0.022;
  ready.lHandX = 0.18 + poise * 0.02;
  ready.lHandY = -0.45;
  ready.lHandZ = -0.15;

  // Spear poised forward & gently bobbing with wrist balance
  ready.rShoulderX = 0.16 + breath * 0.026 - poise * 0.02;
  ready.rShoulderY = -0.06 + weightShift * 0.02;
  ready.rShoulderZ = -0.1 + breath * 0.012;
  ready.rElbowX = -0.82 - breath * 0.026;
  ready.rHandX = 0.36 + poise * 0.032;
  ready.capeLift = 0.13 + breath * 0.032;

  if (mode === 'ready') {
    return ready;
  }

  const pose = createZeroCustomPose();

  if (mode === 'guard') {
    const gPulse = Math.sin(elapsed * 2.6);
    pose.bodyYaw = -0.12 + weightShift * 0.025;
    pose.bodyPitch = 0.06 + gPulse * 0.012;
    pose.bodyRoll = weightShift * 0.02;
    pose.bodyBob = -0.06 + gPulse * 0.012;
    pose.torsoYaw = 0.36;
    pose.torsoPitch = 0.06 + gPulse * 0.018;
    pose.headYaw = -0.32 + Math.sin(elapsed * 0.9) * 0.04;
    pose.headPitch = 0.04;

    pose.lHipX = -0.26;
    pose.lHipZ = 0.08;
    pose.lKneeX = 0.42 - gPulse * 0.025;
    pose.rHipX = 0.24;
    pose.rHipZ = -0.08;
    pose.rKneeX = 0.34 - gPulse * 0.025;

    pose.lShoulderX = -0.94 + gPulse * 0.025;
    pose.lShoulderY = -1.14;
    pose.lShoulderZ = 0;
    pose.lElbowX = -0.48 - gPulse * 0.02;
    pose.lHandX = 0.92;
    pose.lHandY = -0.84;
    pose.lHandZ = -0.9;

    pose.rShoulderX = 0.3 + gPulse * 0.025;
    pose.rShoulderY = -0.14;
    pose.rShoulderZ = -0.14;
    pose.rElbowX = -1.24;
    pose.rHandX = 0.48 - gPulse * 0.02;
    pose.capeLift = 0.15 + gPulse * 0.03;
    return pose;
  }

  if (mode === 'dash') {
    pose.torsoYaw = 0.14;
    pose.headYaw = -0.14;
    pose.lShoulderX = -0.78;
    pose.lShoulderY = -0.92;
    pose.lElbowX = -0.66;
    pose.lHandX = 0.56;
    pose.lHandY = -0.56;
    pose.rShoulderX = 0.54;
    pose.rElbowX = -0.76;
    pose.rHandX = -0.26;
    const d = applyFluidDashKinematics(pose, { dashProgress, dashForward, dashRight });
    if (d.isLateral) {
      if (d.side > 0) {
        // Dashing RIGHT: shield sweeps inward over chest, spear sweeps behind for counter-thrust
        pose.lShoulderX = -0.92;
        pose.lShoulderY = -1.08;
        pose.rShoulderX = 0.66 * d.strength;
        pose.rShoulderZ = -0.28;
        pose.rElbowX = -0.64;
      } else {
        // Dashing LEFT: shield flares outward in phalanx guard on leading left flank, spear coils at hip
        pose.lShoulderX = -1.04;
        pose.lShoulderY = -1.24;
        pose.lShoulderZ = 0.16;
        pose.rShoulderX = 0.38;
        pose.rShoulderY = 0.22;
        pose.rElbowX = -1.02;
      }
    }
    return pose;
  }

  if (mode === 'jump') {
    const rise = THREE.MathUtils.clamp(velY / 7.5, -1, 1);
    const launchLift = Math.max(0, rise);
    const fallBrace = Math.max(0, -rise);
    pose.bodyYaw = -0.1;
    pose.torsoYaw = 0.18;
    pose.torsoPitch = 0.04 - launchLift * 0.08 + fallBrace * 0.1;
    pose.headYaw = -0.16;
    pose.headPitch = rise < -0.2 ? 0.14 : -0.06;

    pose.lShoulderX = -0.74 - launchLift * 0.22 + fallBrace * 0.14;
    pose.lShoulderY = -0.84;
    pose.lShoulderZ = fallBrace * 0.14;
    pose.lElbowX = -0.68 - launchLift * 0.1;
    pose.lHandX = 0.42;
    pose.lHandY = -0.52;

    pose.rShoulderX = 0.36 - launchLift * 0.28 + fallBrace * 0.18;
    pose.rShoulderZ = -0.16 - fallBrace * 0.12;
    pose.rElbowX = -1.22 + fallBrace * 0.16;
    pose.rHandX = 0.62;
    applyFluidAirborneLegs(pose, velY, elapsed, strafeRatio);
    return pose;
  }

  if (mode === 'air_strike') {
    const p = THREE.MathUtils.clamp(strikeProgress, 0, 1);
    const jumpRef = buildCustomLancePose({ mode: 'jump', elapsed, velY, strafeRatio });

    const windup = createZeroCustomPose();
    windup.bodyYaw = 0.18;
    windup.bodyPitch = -0.18;
    windup.bodyBob = 0.07;
    windup.torsoYaw = -0.38;
    windup.torsoPitch = -0.24;
    windup.headYaw = 0.34;
    windup.headPitch = 0.2;
    windup.lHipX = -0.88;
    windup.lHipZ = 0.09;
    windup.lKneeX = 1.3;
    windup.rHipX = -0.28;
    windup.rHipZ = -0.08;
    windup.rKneeX = 0.8;
    windup.lShoulderX = -0.95;
    windup.lShoulderY = -0.95;
    windup.lElbowX = -0.68;
    windup.lHandX = 0.55;
    windup.lHandY = -0.6;
    windup.rShoulderX = -2.35;
    windup.rShoulderY = 0.08;
    windup.rShoulderZ = -0.08;
    windup.rElbowX = -0.35;
    windup.rHandX = 3.02;
    windup.capeLift = 0.65;

    const dive = createZeroCustomPose();
    dive.bodyYaw = -0.16;
    dive.bodyPitch = 0.32;
    dive.bodyBob = -0.04;
    dive.torsoYaw = 0.26;
    dive.torsoPitch = 0.18;
    dive.headYaw = -0.22;
    dive.headPitch = -0.12;
    dive.lHipX = -0.46;
    dive.lHipZ = 0.08;
    dive.lKneeX = 0.48;
    dive.rHipX = 0.58;
    dive.rHipZ = -0.08;
    dive.rKneeX = 0.62;
    dive.lShoulderX = -0.32;
    dive.lShoulderY = -0.45;
    dive.lElbowX = -0.58;
    dive.lHandY = -0.35;
    dive.rShoulderX = -1.78;
    dive.rShoulderY = 0.06;
    dive.rShoulderZ = -0.04;
    dive.rElbowX = -0.04;
    dive.rHandX = 1.7;
    dive.capeLift = 1.2;

    const follow = { ...dive };
    follow.bodyPitch = 0.36;
    follow.bodyBob = -0.07;
    follow.torsoYaw = 0.32;
    follow.torsoPitch = 0.22;
    follow.lShoulderX = -0.22;
    follow.rShoulderX = -1.68;
    follow.rHandX = 1.6;
    follow.capeLift = 1.05;

    return blendFourStageMotion(jumpRef, windup, dive, follow, p, 0.26, 0.54, 0.72);
  }

  if (mode === 'run') {
    const s = Math.min(1.4, speedRatio);
    pose.torsoYaw = 0.14;
    pose.headYaw = -0.12;
    applyFluidGait(pose, {
      stridePhase,
      speedRatio: s,
      backward,
      strafeRatio,
      baseYaw: -0.06,
      leanScale: 1,
    });

    const armPump = Math.sin(stridePhase) * 0.2 * s;
    const armBounce = Math.cos(stridePhase * 2) * 0.05 * s;
    pose.lShoulderX = -0.54 - armPump + armBounce;
    pose.lShoulderY = -0.58;
    pose.lElbowX = -0.68 - Math.max(0, armPump) * 0.25;
    pose.lHandX = 0.22;
    pose.lHandY = -0.42;

    pose.rShoulderX = 0.22 + armPump * 0.85 + armBounce;
    pose.rShoulderZ = -0.12;
    pose.rElbowX = -0.92 - Math.abs(armPump) * 0.18;
    pose.rHandX = 0.38 - armPump * 0.42;
    return pose;
  }

  if (mode === 'strike') {
    const p = THREE.MathUtils.clamp(strikeProgress, 0, 1);

    if (isSpecial) {
      // Lightning Strike ("Удар Грозы"): skyward charge (0..68%) -> gameplay spear-pointing aim & thunder shot (68..92%)
      if (p >= 0.66) {
        const aimIn = THREE.MathUtils.smoothstep(p, 0.66, 0.74);
        const recoil =
          p >= 0.77 && p <= 0.9
            ? Math.sin(((p - 0.77) / 0.13) * Math.PI)
            : 0;
        const rec = THREE.MathUtils.smoothstep(p, 0.89, 1.0);

        const pointPose = createZeroCustomPose();
        pointPose.bodyYaw = -0.18;
        pointPose.bodyPitch = 0.08 - recoil * 0.1;
        pointPose.bodyBob = -0.06;
        pointPose.torsoYaw = 0.34 - recoil * 0.08;
        pointPose.torsoPitch = 0.06 - recoil * 0.08;
        pointPose.headYaw = -0.3;
        pointPose.headPitch = -0.04;
        pointPose.lHipX = -0.44;
        pointPose.lHipZ = 0.09;
        pointPose.lKneeX = 0.62;
        pointPose.lFootX = -0.14;
        pointPose.rHipX = 0.38;
        pointPose.rHipZ = -0.09;
        pointPose.rKneeX = 0.24;
        pointPose.rFootX = -0.18;
        pointPose.lShoulderX = -0.68;
        pointPose.lShoulderY = -0.82;
        pointPose.lElbowX = -0.78;
        pointPose.lHandY = -0.45;
        pointPose.rShoulderX = -1.62 - recoil * 0.14;
        pointPose.rShoulderY = 0.06;
        pointPose.rShoulderZ = -0.04;
        pointPose.rElbowX = -0.05;
        pointPose.rHandX = 1.56 - recoil * 0.08;
        pointPose.capeLift = 0.65 + recoil * 0.55;

        const base = lerpPose(ready, pointPose, aimIn);
        return rec > 0 ? lerpPose(base, ready, rec) : base;
      }

      const charge = createZeroCustomPose();
      charge.bodyYaw = 0.28;
      charge.bodyPitch = -0.22;
      charge.bodyBob = 0.14;
      charge.torsoYaw = -0.52;
      charge.torsoPitch = -0.32;
      charge.headYaw = 0.42;
      charge.headPitch = -0.22;
      charge.lHipX = -0.45;
      charge.lKneeX = 0.72;
      charge.rHipX = 0.22;
      charge.rKneeX = 0.42;
      charge.lShoulderX = -1.08;
      charge.lShoulderY = -1.05;
      charge.lElbowX = -0.68;
      charge.lHandX = 0.62;
      charge.lHandY = -0.66;
      charge.rShoulderX = -2.72;
      charge.rShoulderY = 0.12;
      charge.rElbowX = -0.22;
      charge.rHandX = 2.85;
      charge.capeLift = 0.92;

      return lerpPose(ready, charge, smoothstep01(p / 0.66));
    }

    if (strikeVariant === 3) {
      // 3. Heroic Overhand Achilles Spear Drive
      const windup = createZeroCustomPose();
      windup.bodyYaw = 0.2;
      windup.bodyPitch = -0.14;
      windup.bodyRoll = -0.06;
      windup.bodyBob = 0.1;
      windup.torsoYaw = -0.38;
      windup.torsoPitch = -0.24;
      windup.headYaw = 0.34;
      windup.headPitch = 0.18;
      windup.lHipX = -0.26;
      windup.lKneeX = 0.46;
      windup.rHipX = 0.14;
      windup.rKneeX = 0.34;
      windup.lShoulderX = -0.94;
      windup.lShoulderY = -0.96;
      windup.lElbowX = -0.72;
      windup.lHandX = 0.55;
      windup.lHandY = -0.62;
      windup.rShoulderX = -2.35;
      windup.rShoulderY = 0.08;
      windup.rShoulderZ = -0.06;
      windup.rElbowX = -0.35;
      windup.rHandX = 2.96;
      windup.capeLift = 0.55;

      const strike = createZeroCustomPose();
      strike.bodyYaw = -0.18;
      strike.bodyPitch = 0.2;
      strike.bodyRoll = 0.06;
      strike.bodyBob = -0.08;
      strike.torsoYaw = 0.28;
      strike.torsoPitch = 0.14;
      strike.headYaw = -0.24;
      strike.headPitch = -0.08;
      strike.lHipX = -0.66;
      strike.lHipZ = 0.08;
      strike.lKneeX = 0.9;
      strike.lFootX = -0.22;
      strike.rHipX = 0.52;
      strike.rHipZ = -0.08;
      strike.rKneeX = 0.16;
      strike.rFootX = -0.24;
      strike.lShoulderX = -0.24;
      strike.lShoulderY = -0.35;
      strike.lElbowX = -0.58;
      strike.lHandY = -0.32;
      strike.rShoulderX = -1.94;
      strike.rShoulderY = 0.08;
      strike.rShoulderZ = -0.04;
      strike.rElbowX = -0.04;
      strike.rHandX = 1.66;
      strike.capeLift = 1.15;

      const follow = { ...strike };
      follow.bodyBob = -0.11;
      follow.bodyPitch = 0.24;
      follow.torsoYaw = 0.36;
      follow.torsoPitch = 0.19;
      follow.lKneeX = 0.98;
      follow.lShoulderX = -0.14;
      follow.rShoulderX = -1.86;
      follow.rHandX = 1.56;
      follow.capeLift = 0.92;

      return blendFourStageMotion(ready, windup, strike, follow, p, 0.26, 0.50, 0.69);
    }

    if (strikeVariant === 2) {
      // 2. Shield-Feint into Rising Diagonal Cross-Thrust
      const windup = createZeroCustomPose();
      windup.bodyYaw = 0.2;
      windup.bodyPitch = 0.05;
      windup.bodyRoll = -0.07;
      windup.bodyBob = -0.09;
      windup.torsoYaw = -0.62;
      windup.torsoPitch = 0.08;
      windup.headYaw = 0.56;
      windup.headPitch = -0.04;
      windup.lHipX = -0.3;
      windup.lKneeX = 0.48;
      windup.rHipX = 0.3;
      windup.rKneeX = 0.4;
      windup.lShoulderX = -0.98;
      windup.lShoulderY = -0.62;
      windup.lElbowX = -0.58;
      windup.lHandX = 0.35;
      windup.lHandY = -0.48;
      windup.rShoulderX = 0.4;
      windup.rShoulderY = -0.14;
      windup.rShoulderZ = -0.22;
      windup.rElbowX = -1.26;
      windup.rHandX = 0.74;
      windup.capeLift = 0.32;

      const strike = createZeroCustomPose();
      strike.bodyYaw = -0.18;
      strike.bodyPitch = 0.15;
      strike.bodyRoll = 0.06;
      strike.bodyBob = -0.07;
      strike.torsoYaw = 0.38;
      strike.torsoPitch = 0.02;
      strike.headYaw = -0.34;
      strike.headPitch = -0.02;
      strike.lHipX = -0.56;
      strike.lHipZ = 0.08;
      strike.lKneeX = 0.74;
      strike.lFootX = -0.2;
      strike.rHipX = 0.48;
      strike.rHipZ = -0.08;
      strike.rKneeX = 0.2;
      strike.rFootX = -0.22;
      strike.lShoulderX = -0.22;
      strike.lShoulderY = -0.28;
      strike.lElbowX = -0.52;
      strike.lHandY = -0.22;
      strike.rShoulderX = -1.92;
      strike.rShoulderY = 0.08;
      strike.rShoulderZ = 0.1;
      strike.rElbowX = -0.05;
      strike.rHandX = 1.62;
      strike.capeLift = 0.94;

      const follow = { ...strike };
      follow.bodyBob = -0.1;
      follow.torsoYaw = 0.45;
      follow.rShoulderX = -1.98;
      follow.rHandX = 1.56;
      follow.lShoulderX = -0.14;
      follow.capeLift = 0.78;

      return blendFourStageMotion(ready, windup, strike, follow, p, 0.24, 0.48, 0.66);
    }

    // 1. Classic Hoplite Shield-Locked Horizontal Piercing Spear Lunge
    const windup = createZeroCustomPose();
    windup.bodyYaw = 0.22;
    windup.bodyPitch = -0.05;
    windup.bodyRoll = -0.06;
    windup.bodyBob = -0.09;
    windup.torsoYaw = -0.58;
    windup.torsoPitch = -0.04;
    windup.headYaw = 0.54;
    windup.headPitch = 0.04;
    windup.lHipX = -0.2;
    windup.lHipZ = 0.07;
    windup.lKneeX = 0.36;
    windup.rHipX = 0.28;
    windup.rHipZ = -0.08;
    windup.rKneeX = 0.44;
    windup.lShoulderX = -0.9;
    windup.lShoulderY = -1.06;
    windup.lElbowX = -0.72;
    windup.lHandX = 0.58;
    windup.lHandY = -0.68;
    windup.rShoulderX = 0.58;
    windup.rShoulderY = -0.04;
    windup.rShoulderZ = -0.1;
    windup.rElbowX = -1.54;
    windup.rHandX = 0.96;
    windup.capeLift = 0.28;

    const strike = createZeroCustomPose();
    strike.bodyYaw = -0.2;
    strike.bodyPitch = 0.19;
    strike.bodyRoll = 0.05;
    strike.bodyBob = -0.09;
    strike.torsoYaw = 0.36;
    strike.torsoPitch = 0.08;
    strike.headYaw = -0.32;
    strike.headPitch = -0.06;
    strike.lHipX = -0.66;
    strike.lHipZ = 0.08;
    strike.lKneeX = 0.9;
    strike.lFootX = -0.22;
    strike.rHipX = 0.54;
    strike.rHipZ = -0.08;
    strike.rKneeX = 0.14;
    strike.rFootX = -0.24;
    strike.lShoulderX = -0.18;
    strike.lShoulderY = -0.26;
    strike.lElbowX = -0.55;
    strike.lHandY = -0.2;
    strike.rShoulderX = -1.9;
    strike.rShoulderY = 0.04;
    strike.rShoulderZ = -0.04;
    strike.rElbowX = -0.05;
    strike.rHandX = 1.66;
    strike.capeLift = 0.96;

    const follow = { ...strike };
    follow.bodyBob = -0.115;
    follow.bodyPitch = 0.23;
    follow.torsoYaw = 0.44;
    follow.lKneeX = 0.96;
    follow.rShoulderX = -1.84;
    follow.rHandX = 1.56;
    follow.capeLift = 0.8;

    return blendFourStageMotion(ready, windup, strike, follow, p, 0.24, 0.48, 0.65);
  }

  return ready;
}

export function buildCustomMeleePose(params: {
  mode: 'ready' | 'run' | 'dash' | 'guard' | 'jump' | 'strike' | 'air_strike';
  elapsed?: number;
  speedRatio?: number;
  stridePhase?: number;
  strikeProgress?: number;
  strikeVariant?: number;
  velY?: number;
  backward?: boolean;
  strafeRatio?: number;
  dashProgress?: number;
  dashForward?: number;
  dashRight?: number;
  isHeavy?: boolean;
  isTwinBlade?: boolean;
  isWindSpecial?: boolean;
  isHammerSpecial?: boolean;
}): CustomBonePose {
  const {
    mode,
    elapsed = 0,
    speedRatio = 0,
    stridePhase = 0,
    strikeProgress = 0,
    strikeVariant = 1,
    velY = 0,
    backward = false,
    strafeRatio = 0,
    dashProgress = 0.5,
    dashForward = 1,
    dashRight = 0,
    isHeavy = false,
    isTwinBlade = false,
    isWindSpecial = false,
    isHammerSpecial = false,
  } = params;

  const breath = Math.sin(elapsed * 2.3) + 0.25 * Math.sin(elapsed * 4.6 + 0.4);
  const bounce = Math.sin(elapsed * 3.15);
  const weightShift = Math.sin(elapsed * 1.05);
  const wristPoise = Math.sin(elapsed * 1.65 + 0.9);
  const ready = createZeroCustomPose();
  if (isTwinBlade) {
    // Coiled agile shinobi stance with subtle spring bounce, hip shift & twin-kunai poise
    ready.bodyYaw = -0.18 + weightShift * 0.048;
    ready.bodyPitch = 0.08 + breath * 0.015;
    ready.bodyRoll = weightShift * 0.032;
    ready.bodyBob = -0.012 + bounce * 0.018 + breath * 0.008;
    ready.torsoYaw = 0.34 - weightShift * 0.055;
    ready.torsoPitch = 0.12 + breath * 0.025;
    ready.torsoRoll = -0.04 - weightShift * 0.028;
    ready.headYaw = -0.28 + Math.sin(elapsed * 0.85) * 0.06;
    ready.headPitch = -0.1 - breath * 0.015;

    ready.lHipX = -0.26 - bounce * 0.028 - weightShift * 0.025;
    ready.lHipZ = 0.09;
    ready.lKneeX = 0.44 + bounce * 0.055 + weightShift * 0.045;
    ready.lFootX = -0.2 - bounce * 0.025;
    ready.rHipX = 0.22 - bounce * 0.024 + weightShift * 0.025;
    ready.rHipZ = -0.09;
    ready.rKneeX = 0.36 + bounce * 0.05 - weightShift * 0.045;
    ready.rFootX = -0.34 - bounce * 0.022;

    ready.lShoulderX = -0.82 + breath * 0.032 + wristPoise * 0.025;
    ready.lShoulderY = -0.42;
    ready.lShoulderZ = 0.24 + breath * 0.018;
    ready.lElbowX = -1.16 - bounce * 0.035;
    ready.lHandX = 0.18 + wristPoise * 0.045;
    ready.lHandY = -0.35;
    ready.lWeaponX = wristPoise * 0.05;

    ready.rShoulderX = 0.24 - breath * 0.032 - wristPoise * 0.025;
    ready.rShoulderY = 0.18;
    ready.rShoulderZ = -0.28 - breath * 0.018;
    ready.rElbowX = -0.94 - bounce * 0.035;
    ready.rHandX = 0.22 - wristPoise * 0.045;
    ready.rWeaponX = -wristPoise * 0.05;
    ready.capeLift = 0.18 + breath * 0.04;
  } else if (isHeavy) {
    // Heroic Xenon stance: wide grounded legs, clenched left fist poised at waist, enlarged hammer held upright beside right shoulder
    ready.bodyYaw = weightShift * 0.03;
    ready.bodyRoll = weightShift * 0.014;
    ready.bodyBob = -0.025 + breath * 0.012;
    ready.torsoPitch = 0.04 + breath * 0.014;
    ready.torsoYaw = -0.22 - weightShift * 0.03;
    ready.headYaw = 0.2 + Math.sin(elapsed * 0.72) * 0.035;
    ready.headPitch = -0.02;

    ready.lHipX = -0.2 - weightShift * 0.018;
    ready.lHipZ = 0.14;
    ready.lKneeX = 0.32 + weightShift * 0.025;
    ready.lFootX = -0.12;
    ready.rHipX = 0.2 + weightShift * 0.018;
    ready.rHipZ = -0.14;
    ready.rKneeX = 0.24 - weightShift * 0.025;
    ready.rFootX = -0.12;

    ready.rShoulderX = -0.2 + breath * 0.018;
    ready.rShoulderY = -0.12;
    ready.rShoulderZ = -0.44 - breath * 0.01;
    ready.rElbowX = -0.78 - breath * 0.016;
    ready.rHandX = -0.24 + wristPoise * 0.018;
    ready.rHandZ = -0.08;

    ready.lShoulderX = -0.16 + breath * 0.018;
    ready.lShoulderY = -0.16;
    ready.lShoulderZ = 0.32;
    ready.lElbowX = -0.82 - breath * 0.016;
    ready.lHandX = 0.12;
    ready.lHandZ = -0.06;
    ready.capeLift = 0.18 + breath * 0.032;
  } else {
    ready.bodyYaw = weightShift * 0.035;
    ready.bodyRoll = weightShift * 0.025;
    ready.bodyBob = breath * 0.018;
    ready.torsoPitch = 0.04 + breath * 0.02;
    ready.torsoYaw = 0.1 - weightShift * 0.04;
    ready.headYaw = -0.07 + Math.sin(elapsed * 0.8) * 0.05;
    ready.lHipX = -0.14 - weightShift * 0.02;
    ready.lHipZ = 0.06;
    ready.lKneeX = 0.24 + weightShift * 0.04;
    ready.rHipX = 0.14 + weightShift * 0.02;
    ready.rHipZ = -0.06;
    ready.rKneeX = 0.2 - weightShift * 0.04;
    ready.lShoulderX = -0.35 + breath * 0.025;
    ready.lShoulderZ = 0.18;
    ready.lElbowX = -0.58 - breath * 0.02;
    ready.rShoulderX = -0.42 + breath * 0.025;
    ready.rShoulderZ = -0.18;
    ready.rElbowX = -0.62 - breath * 0.02;
    ready.capeLift = 0.12 + breath * 0.03;
  }

  if (mode === 'ready') return ready;

  const pose = createZeroCustomPose();

  if (mode === 'run') {
    const s = Math.min(1.4, speedRatio);
    pose.torsoPitch = isTwinBlade ? 0.22 : isHeavy ? 0.12 : 0.06;
    pose.torsoYaw = isHeavy ? -0.18 : 0;
    pose.headPitch = isTwinBlade ? -0.16 : isHeavy ? -0.06 : 0;
    pose.headYaw = isHeavy ? 0.16 : 0;
    applyFluidGait(pose, {
      stridePhase,
      speedRatio: s,
      backward,
      strafeRatio,
      leanScale: isTwinBlade ? 1.38 : isHeavy ? 1.08 : 1,
    });
    const armDrive = Math.sin(stridePhase) * 0.46 * s;
    const armFlutter = Math.cos(stridePhase * 2) * 0.065 * s;
    if (isTwinBlade) {
      // Iconic Kunoichi ninja dash-sprint with fluid shoulder sweep & wrist flutter
      pose.rShoulderX = 0.68 * Math.min(1.15, s) + armDrive * 0.28 + armFlutter;
      pose.lShoulderX = 0.68 * Math.min(1.15, s) - armDrive * 0.28 + armFlutter;
      pose.rShoulderZ = -0.34 - Math.abs(armDrive) * 0.08;
      pose.lShoulderZ = 0.34 + Math.abs(armDrive) * 0.08;
      pose.rElbowX = -0.34 - armFlutter * 0.8;
      pose.lElbowX = -0.34 - armFlutter * 0.8;
      pose.rHandX = 0.18 + armDrive * 0.12;
      pose.lHandX = 0.18 - armDrive * 0.12;
    } else if (isHeavy) {
      // Grounded armored march carrying enlarged hammer upright clear of legs
      pose.rShoulderX = -0.34 + armDrive * 0.24 + armFlutter * 0.4;
      pose.rShoulderY = -0.1;
      pose.rShoulderZ = -0.34;
      pose.rElbowX = -0.82 - Math.max(0, -armDrive) * 0.16;
      pose.rHandX = 0.08 - armDrive * 0.08;
      pose.rHandZ = -0.1;

      pose.lShoulderX = -0.28 - armDrive * 0.56;
      pose.lShoulderZ = 0.28;
      pose.lElbowX = -0.78 - Math.max(0, armDrive) * 0.22;
      pose.capeLift = Math.max(pose.capeLift, 0.24 + s * 0.42);
    } else {
      pose.rShoulderX = -0.24 + armDrive;
      pose.lShoulderX = -0.24 - armDrive;
      pose.rShoulderZ = -0.18;
      pose.lShoulderZ = 0.18;
      pose.rElbowX = -0.55 - Math.max(0, -armDrive) * 0.25;
      pose.lElbowX = -0.55 - Math.max(0, armDrive) * 0.25;
    }
    return pose;
  }

  if (mode === 'dash') {
    if (isTwinBlade) {
      pose.torsoPitch = 0.24;
      pose.headPitch = -0.18;
      pose.rShoulderX = 0.84;
      pose.lShoulderX = 0.84;
      pose.rShoulderZ = -0.38;
      pose.lShoulderZ = 0.38;
      pose.rElbowX = -0.22;
      pose.lElbowX = -0.22;
    } else if (isHeavy) {
      pose.torsoPitch = 0.22;
      pose.torsoYaw = -0.2;
      pose.headYaw = 0.18;
      pose.headPitch = -0.12;
      pose.rShoulderX = 0.36;
      pose.rShoulderZ = -0.28;
      pose.rElbowX = -0.52;
      pose.rHandX = 0.22;
      pose.rHandZ = -0.14;
      pose.lShoulderX = -0.88;
      pose.lShoulderZ = 0.16;
      pose.lElbowX = -0.82;
    } else {
      pose.torsoPitch = 0.16;
      pose.rShoulderX = 0.56;
      pose.lShoulderX = 0.56;
      pose.rShoulderZ = -0.26;
      pose.lShoulderZ = 0.26;
      pose.rElbowX = -0.32;
      pose.lElbowX = -0.32;
    }
    const d = applyFluidDashKinematics(pose, { dashProgress, dashForward, dashRight });
    if (d.isLateral) {
      if (isTwinBlade) {
        // Kunoichi left / right lateral shinobi dash-skate
        pose.bodyRoll = d.side * 0.44 * d.strength;
        if (d.side > 0) {
          // Right dash: right blade tucked cross-guard, left blade trailing behind like wing
          pose.rShoulderX = -0.94;
          pose.rShoulderZ = 0.38;
          pose.rElbowX = -1.06;
          pose.lShoulderX = 0.72 * d.strength;
          pose.lShoulderZ = 0.48;
          pose.lElbowX = -0.26;
        } else {
          // Left dash: left blade tucked cross-guard, right blade trailing behind like wing
          pose.lShoulderX = -0.94;
          pose.lShoulderZ = -0.38;
          pose.lElbowX = -1.06;
          pose.rShoulderX = 0.72 * d.strength;
          pose.rShoulderZ = -0.48;
          pose.rElbowX = -0.26;
        }
      } else if (isHeavy) {
        // Xenon left / right lateral armored thruster skid
        if (d.side > 0) {
          // Right dash: warhammer slung out right in wake, left gauntlet braced inward
          pose.rShoulderX = -0.38;
          pose.rShoulderZ = -0.58 * d.strength;
          pose.rElbowX = -0.44;
          pose.lShoulderX = -0.98;
          pose.lShoulderZ = -0.18;
          pose.lElbowX = -0.9;
        } else {
          // Left dash: hammer coiled close across right torso, left gauntlet flared wide for balance
          pose.rShoulderX = -0.78;
          pose.rShoulderZ = 0.22 * d.strength;
          pose.rElbowX = -0.92;
          pose.lShoulderX = -0.42;
          pose.lShoulderZ = 0.56 * d.strength;
          pose.lElbowX = -0.48;
        }
      } else {
        // Odin / Halberd left vs right lateral thruster skate
        if (d.side > 0) {
          pose.rShoulderX = 0.68 * d.strength;
          pose.rShoulderZ = -0.36;
          pose.lShoulderX = -0.82;
          pose.lShoulderZ = 0.24;
        } else {
          pose.lShoulderX = 0.68 * d.strength;
          pose.lShoulderZ = 0.36;
          pose.rShoulderX = -0.82;
          pose.rShoulderZ = -0.24;
        }
      }
    }
    return pose;
  }

  if (mode === 'guard') {
    const gPulse = Math.sin(elapsed * 2.9);
    if (isTwinBlade) {
      pose.bodyPitch = 0.08 + gPulse * 0.012;
      pose.bodyRoll = weightShift * 0.025;
      pose.bodyBob = -0.018 + gPulse * 0.014;
      pose.torsoPitch = 0.1 + gPulse * 0.018;
      pose.headPitch = -0.08;
      pose.lHipX = -0.28;
      pose.lHipZ = 0.1;
      pose.lKneeX = 0.48 - gPulse * 0.03;
      pose.lFootX = -0.22;
      pose.rHipX = 0.24;
      pose.rHipZ = -0.1;
      pose.rKneeX = 0.42 - gPulse * 0.03;
      pose.rFootX = -0.36;
      pose.lShoulderX = -1.16 + gPulse * 0.028;
      pose.lShoulderY = -0.38;
      pose.lShoulderZ = -0.42;
      pose.lElbowX = -1.06 - gPulse * 0.025;
      pose.rShoulderX = -1.1 + gPulse * 0.028;
      pose.rShoulderY = 0.38;
      pose.rShoulderZ = 0.42;
      pose.rElbowX = -1.02 - gPulse * 0.025;
      pose.rHandX = 0.28 + gPulse * 0.025;
      pose.lHandX = 0.28 - gPulse * 0.025;
      pose.capeLift = 0.2 + gPulse * 0.03;
      return pose;
    }
    if (isHeavy) {
      // Planted iron cross-chest warhammer block braced by the left gauntlet
      pose.bodyPitch = 0.05 + gPulse * 0.01;
      pose.bodyBob = -0.06 + gPulse * 0.01;
      pose.lHipX = -0.26;
      pose.lHipZ = 0.1;
      pose.rHipX = 0.24;
      pose.rHipZ = -0.1;
      pose.lKneeX = 0.42;
      pose.rKneeX = 0.38;
      pose.torsoYaw = 0.18;
      pose.headYaw = -0.16;
      pose.rShoulderX = -1.02 + gPulse * 0.02;
      pose.rShoulderZ = 0.18;
      pose.rElbowX = -0.82;
      pose.rHandX = 0.18;
      pose.rHandZ = 0.24;
      pose.lShoulderX = -0.96 + gPulse * 0.02;
      pose.lShoulderY = -0.32;
      pose.lShoulderZ = -0.18;
      pose.lElbowX = -0.95;
      pose.capeLift = 0.22 + gPulse * 0.03;
      return pose;
    }
    pose.bodyPitch = 0.05 + gPulse * 0.012;
    pose.bodyBob = -0.05 + gPulse * 0.012;
    pose.lHipX = -0.22;
    pose.rHipX = 0.22;
    pose.lKneeX = 0.36;
    pose.rKneeX = 0.34;
    pose.torsoYaw = 0.34;
    pose.headYaw = -0.3;
    pose.lShoulderX = -1.08 + gPulse * 0.025;
    pose.lShoulderZ = -0.34;
    pose.lElbowX = -0.68;
    pose.rShoulderX = -0.45 + gPulse * 0.02;
    pose.rElbowX = -0.62;
    pose.capeLift = 0.16;
    return pose;
  }

  if (mode === 'jump') {
    const rise = THREE.MathUtils.clamp(velY / 7.5, -1, 1);
    const launchLift = Math.max(0, rise);
    const fallBrace = Math.max(0, -rise);
    if (isTwinBlade) {
      pose.torsoPitch = 0.14 - launchLift * 0.08 + fallBrace * 0.12;
      pose.headPitch = rise < -0.2 ? 0.1 : -0.1;
      pose.lShoulderX = 0.42 - launchLift * 0.52 + fallBrace * 0.25;
      pose.lShoulderZ = 0.46 + fallBrace * 0.18;
      pose.lElbowX = -0.68;
      pose.rShoulderX = -0.76 - launchLift * 0.28 + fallBrace * 0.2;
      pose.rShoulderZ = -0.32 - fallBrace * 0.16;
      pose.rElbowX = -1.02;
    } else if (isHeavy) {
      pose.torsoPitch = 0.08 - launchLift * 0.06 + fallBrace * 0.1;
      pose.torsoYaw = -0.18;
      pose.headYaw = 0.16;
      pose.headPitch = rise < -0.2 ? 0.12 : -0.04;
      pose.rShoulderX = -0.42 - launchLift * 0.28 + fallBrace * 0.18;
      pose.rShoulderZ = -0.36 - fallBrace * 0.12;
      pose.rElbowX = -0.82;
      pose.rHandX = 0.08;
      pose.rHandZ = -0.1;
      pose.lShoulderX = -0.44 - launchLift * 0.26 + fallBrace * 0.22;
      pose.lShoulderZ = 0.34 + fallBrace * 0.18;
      pose.lElbowX = -0.78;
    } else {
      pose.lShoulderX = -0.65 - launchLift * 0.28 + fallBrace * 0.22;
      pose.lShoulderZ = 0.28 + fallBrace * 0.18;
      pose.lElbowX = -0.58;
      pose.rShoulderX = -0.48 - launchLift * 0.26 + fallBrace * 0.18;
      pose.rShoulderZ = -0.26 - fallBrace * 0.16;
      pose.rElbowX = -0.62;
    }
    applyFluidAirborneLegs(pose, velY, elapsed, strafeRatio);
    return pose;
  }

  if (mode === 'air_strike') {
    return buildCustomJumpPose({
      mode: 'air_strike',
      velY,
      strikeProgress,
      isHeavy,
      isTwinBlade,
      elapsed,
      strafeRatio,
    });
  }

  const p = THREE.MathUtils.clamp(strikeProgress, 0, 1);

  // Signature ultra-realistic martial Shinobi Combo + "Быстрый как ветер" Superattack choreography
  if (isTwinBlade) {
    if (isWindSpecial) {
      // "Быстрый как ветер": low-center shinobi coil -> 3 explosive hip-driven horizontal cyclone cuts ->
      // deep cat-crouch gather -> rising bottom-to-top dragon-fang uppercut -> cushioned landing recovery
      const coil = createZeroCustomPose();
      coil.bodyYaw = -0.34;
      coil.bodyPitch = 0.22;
      coil.bodyRoll = -0.1;
      coil.bodyBob = -0.11;
      coil.torsoYaw = -0.78;
      coil.torsoPitch = 0.2;
      coil.torsoRoll = -0.1;
      coil.headYaw = 0.72;
      coil.headPitch = -0.14;
      coil.lHipX = -0.46;
      coil.lHipZ = 0.14;
      coil.lKneeX = 0.84;
      coil.lFootX = -0.28;
      coil.rHipX = 0.36;
      coil.rHipZ = -0.14;
      coil.rKneeX = 0.68;
      coil.rFootX = -0.32;
      coil.rShoulderX = -0.95;
      coil.rShoulderZ = -0.82;
      coil.rElbowX = -0.62;
      coil.rHandY = -0.38;
      coil.rWeaponX = -1.18;
      coil.lShoulderX = 0.42;
      coil.lShoulderZ = 0.56;
      coil.lElbowX = -0.74;
      coil.lHandY = 0.3;
      coil.lWeaponX = -1.25;
      coil.capeLift = 0.64;

      const slash1 = createZeroCustomPose();
      slash1.bodyYaw = 0.42;
      slash1.bodyPitch = 0.24;
      slash1.bodyRoll = 0.14;
      slash1.bodyBob = -0.05;
      slash1.torsoYaw = 0.86;
      slash1.torsoPitch = 0.14;
      slash1.torsoRoll = 0.12;
      slash1.headYaw = -0.76;
      slash1.headPitch = -0.12;
      slash1.lHipX = -0.62;
      slash1.lHipZ = 0.12;
      slash1.lKneeX = 0.86;
      slash1.lFootX = -0.34;
      slash1.rHipX = 0.48;
      slash1.rHipZ = -0.12;
      slash1.rKneeX = 0.24;
      slash1.rFootX = -0.36;
      slash1.rShoulderX = -1.36;
      slash1.rShoulderY = 0.42;
      slash1.rShoulderZ = 0.86;
      slash1.rElbowX = -0.16;
      slash1.rHandY = 0.55;
      slash1.rWeaponX = -1.52;
      slash1.lShoulderX = -1.15;
      slash1.lShoulderY = -0.25;
      slash1.lShoulderZ = -0.44;
      slash1.lElbowX = -0.45;
      slash1.lWeaponX = -1.38;
      slash1.capeLift = 1.08;

      const slash2 = createZeroCustomPose();
      slash2.bodyYaw = -0.48;
      slash2.bodyPitch = 0.25;
      slash2.bodyRoll = -0.16;
      slash2.bodyBob = -0.04;
      slash2.torsoYaw = -0.9;
      slash2.torsoPitch = 0.14;
      slash2.torsoRoll = -0.14;
      slash2.headYaw = 0.82;
      slash2.headPitch = -0.12;
      slash2.lHipX = 0.46;
      slash2.lHipZ = 0.12;
      slash2.lKneeX = 0.26;
      slash2.lFootX = -0.34;
      slash2.rHipX = -0.64;
      slash2.rHipZ = -0.12;
      slash2.rKneeX = 0.88;
      slash2.rFootX = -0.34;
      slash2.rShoulderX = -1.28;
      slash2.rShoulderY = -0.28;
      slash2.rShoulderZ = -0.9;
      slash2.rElbowX = -0.18;
      slash2.rHandY = -0.48;
      slash2.rWeaponX = -1.48;
      slash2.lShoulderX = -1.4;
      slash2.lShoulderY = -0.4;
      slash2.lShoulderZ = 0.88;
      slash2.lElbowX = -0.16;
      slash2.lHandY = -0.52;
      slash2.lWeaponX = -1.54;
      slash2.capeLift = 1.18;

      const cycloneSpin = createZeroCustomPose();
      cycloneSpin.bodyYaw = Math.PI * 2;
      cycloneSpin.bodyPitch = 0.2;
      cycloneSpin.bodyRoll = 0.08;
      cycloneSpin.bodyBob = 0.06;
      cycloneSpin.torsoYaw = 0.38;
      cycloneSpin.torsoPitch = 0.14;
      cycloneSpin.headYaw = -0.32;
      cycloneSpin.lHipX = -0.54;
      cycloneSpin.lKneeX = 0.88;
      cycloneSpin.rHipX = 0.34;
      cycloneSpin.rKneeX = 0.52;
      cycloneSpin.rShoulderX = -1.28;
      cycloneSpin.rShoulderZ = -0.96;
      cycloneSpin.rElbowX = -0.14;
      cycloneSpin.rHandY = 0.35;
      cycloneSpin.rWeaponX = -1.52;
      cycloneSpin.lShoulderX = -1.28;
      cycloneSpin.lShoulderZ = 0.96;
      cycloneSpin.lElbowX = -0.14;
      cycloneSpin.lHandY = -0.35;
      cycloneSpin.lWeaponX = -1.52;
      cycloneSpin.capeLift = 1.32;

      const upperCoil = createZeroCustomPose();
      upperCoil.bodyYaw = Math.PI * 2 + 0.14;
      upperCoil.bodyBob = -0.16;
      upperCoil.bodyPitch = 0.28;
      upperCoil.torsoPitch = 0.28;
      upperCoil.torsoYaw = -0.22;
      upperCoil.headPitch = -0.22;
      upperCoil.lHipX = -0.58;
      upperCoil.lKneeX = 1.02;
      upperCoil.lFootX = -0.32;
      upperCoil.rHipX = 0.44;
      upperCoil.rKneeX = 0.78;
      upperCoil.rFootX = -0.34;
      upperCoil.rShoulderX = 0.86;
      upperCoil.rShoulderZ = -0.36;
      upperCoil.rElbowX = -0.38;
      upperCoil.rHandX = -0.18;
      upperCoil.rWeaponX = -1.16;
      upperCoil.lShoulderX = 0.86;
      upperCoil.lShoulderZ = 0.36;
      upperCoil.lElbowX = -0.38;
      upperCoil.lHandX = -0.18;
      upperCoil.lWeaponX = -1.16;
      upperCoil.capeLift = 0.56;

      const upperLaunch = createZeroCustomPose();
      upperLaunch.bodyYaw = Math.PI * 2 - 0.14;
      upperLaunch.bodyBob = 0.22;
      upperLaunch.bodyPitch = -0.28;
      upperLaunch.torsoPitch = -0.36;
      upperLaunch.torsoYaw = 0.18;
      upperLaunch.headPitch = -0.24;
      upperLaunch.lHipX = -0.86;
      upperLaunch.lKneeX = 1.24;
      upperLaunch.lFootX = -0.3;
      upperLaunch.rHipX = 0.42;
      upperLaunch.rKneeX = 0.24;
      upperLaunch.rFootX = -0.28;
      upperLaunch.rShoulderX = -2.68;
      upperLaunch.rShoulderY = 0.22;
      upperLaunch.rShoulderZ = 0.24;
      upperLaunch.rElbowX = -0.08;
      upperLaunch.rHandX = 0.28;
      upperLaunch.rWeaponX = -1.62;
      upperLaunch.lShoulderX = -2.64;
      upperLaunch.lShoulderY = -0.22;
      upperLaunch.lShoulderZ = -0.24;
      upperLaunch.lElbowX = -0.08;
      upperLaunch.lHandX = 0.28;
      upperLaunch.lWeaponX = -1.62;
      upperLaunch.capeLift = 1.42;

      const upperLand = createZeroCustomPose();
      upperLand.bodyYaw = Math.PI * 2 - 0.08;
      upperLand.bodyBob = -0.11;
      upperLand.bodyPitch = 0.18;
      upperLand.torsoPitch = 0.18;
      upperLand.lHipX = -0.44;
      upperLand.lKneeX = 0.74;
      upperLand.lFootX = -0.26;
      upperLand.rHipX = 0.32;
      upperLand.rKneeX = 0.52;
      upperLand.rFootX = -0.28;
      upperLand.rShoulderX = -0.92;
      upperLand.rShoulderZ = -0.44;
      upperLand.rElbowX = -0.82;
      upperLand.lShoulderX = -0.88;
      upperLand.lShoulderZ = 0.44;
      upperLand.lElbowX = -0.84;
      upperLand.capeLift = 0.65;

      const recoverReady = { ...ready, bodyYaw: Math.PI * 2 + ready.bodyYaw };
      return blendMultiStageMotion(
        [
          { t: 0.0, pose: ready },
          { t: 0.11, pose: coil, ease: 'smooth' },
          { t: 0.26, pose: slash1, ease: 'quart' },
          { t: 0.43, pose: slash2, ease: 'quart' },
          { t: 0.62, pose: cycloneSpin, ease: 'cubic' },
          { t: 0.74, pose: upperCoil, ease: 'smooth' },
          { t: 0.86, pose: upperLaunch, ease: 'quart' },
          { t: 0.93, pose: upperLand, ease: 'smooth' },
          { t: 1.0, pose: recoverReady, ease: 'quad' },
        ],
        p,
      );
    }

    if (strikeVariant === 3) {
      // 3. 360° Spinning Cyclone Twin-Fang Finisher: cat-coil -> corkscrew spin -> cross-slash -> ninja landing absorb
      const windup = createZeroCustomPose();
      windup.bodyYaw = 0.28;
      windup.bodyBob = -0.13;
      windup.bodyPitch = 0.18;
      windup.bodyRoll = -0.09;
      windup.torsoYaw = -0.88;
      windup.torsoPitch = 0.14;
      windup.headYaw = 0.74;
      windup.lHipX = -0.48;
      windup.lHipZ = 0.12;
      windup.lKneeX = 0.84;
      windup.lFootX = -0.28;
      windup.rHipX = 0.36;
      windup.rHipZ = -0.12;
      windup.rKneeX = 0.68;
      windup.rFootX = -0.3;
      windup.rShoulderX = -0.92;
      windup.rShoulderZ = -0.86;
      windup.rElbowX = -0.48;
      windup.rHandY = -0.36;
      windup.lShoulderX = -0.92;
      windup.lShoulderZ = 0.86;
      windup.lElbowX = -0.48;
      windup.lHandY = 0.36;
      windup.capeLift = 0.42;

      const spinMid = createZeroCustomPose();
      spinMid.bodyYaw = Math.PI * 1.15;
      spinMid.bodyBob = 0.16;
      spinMid.bodyPitch = 0.2;
      spinMid.bodyRoll = 0.16;
      spinMid.torsoPitch = 0.12;
      spinMid.lHipX = -0.66;
      spinMid.lKneeX = 1.02;
      spinMid.rHipX = 0.28;
      spinMid.rKneeX = 0.56;
      spinMid.rShoulderX = -1.26;
      spinMid.rShoulderZ = -1.04;
      spinMid.rElbowX = -0.14;
      spinMid.rHandY = 0.42;
      spinMid.rWeaponX = -1.45;
      spinMid.lShoulderX = -1.26;
      spinMid.lShoulderZ = 1.04;
      spinMid.lElbowX = -0.14;
      spinMid.lHandY = -0.42;
      spinMid.lWeaponX = -1.45;
      spinMid.capeLift = 1.25;

      const spinEnd = createZeroCustomPose();
      spinEnd.bodyYaw = Math.PI * 2;
      spinEnd.bodyBob = 0.08;
      spinEnd.bodyPitch = 0.22;
      spinEnd.bodyRoll = -0.1;
      spinEnd.torsoYaw = 0.56;
      spinEnd.torsoPitch = 0.16;
      spinEnd.headYaw = -0.48;
      spinEnd.lHipX = -0.58;
      spinEnd.lKneeX = 0.9;
      spinEnd.rHipX = 0.38;
      spinEnd.rKneeX = 0.44;
      spinEnd.rShoulderX = -1.32;
      spinEnd.rShoulderZ = 0.62;
      spinEnd.rElbowX = -0.18;
      spinEnd.rWeaponX = -1.52;
      spinEnd.lShoulderX = -1.32;
      spinEnd.lShoulderZ = -0.58;
      spinEnd.lElbowX = -0.18;
      spinEnd.lWeaponX = -1.52;
      spinEnd.capeLift = 1.18;

      const land = createZeroCustomPose();
      land.bodyYaw = Math.PI * 2 - 0.16;
      land.bodyBob = -0.12;
      land.bodyPitch = 0.18;
      land.torsoYaw = 0.42;
      land.torsoPitch = 0.16;
      land.headYaw = -0.34;
      land.lHipX = -0.42;
      land.lHipZ = 0.12;
      land.lKneeX = 0.72;
      land.lFootX = -0.26;
      land.rHipX = 0.32;
      land.rHipZ = -0.12;
      land.rKneeX = 0.56;
      land.rFootX = -0.3;
      land.rShoulderX = -0.78;
      land.rShoulderZ = 0.54;
      land.rElbowX = -0.54;
      land.lShoulderX = 0.52;
      land.lShoulderZ = 0.42;
      land.lElbowX = -0.68;
      land.capeLift = 0.7;

      const recoverReady = { ...ready, bodyYaw: Math.PI * 2 + ready.bodyYaw };
      return blendMultiStageMotion(
        [
          { t: 0.0, pose: ready },
          { t: 0.18, pose: windup, ease: 'smooth' },
          { t: 0.42, pose: spinMid, ease: 'cubic' },
          { t: 0.64, pose: spinEnd, ease: 'quart' },
          { t: 0.8, pose: land, ease: 'smooth' },
          { t: 1.0, pose: recoverReady, ease: 'quad' },
        ],
        p,
      );
    }

    if (strikeVariant === 2) {
      // 2. Low-Stance Shinobi Spring Lunge into Crisp Twin-Kunai X-Scissor Cut & Elastic Recoil
      const crouchCoil = createZeroCustomPose();
      crouchCoil.bodyBob = -0.13;
      crouchCoil.bodyPitch = 0.1;
      crouchCoil.torsoPitch = -0.14;
      crouchCoil.torsoYaw = -0.26;
      crouchCoil.headYaw = 0.22;
      crouchCoil.headPitch = -0.12;
      crouchCoil.lHipX = -0.38;
      crouchCoil.lHipZ = 0.12;
      crouchCoil.lKneeX = 0.68;
      crouchCoil.lFootX = -0.26;
      crouchCoil.rHipX = 0.32;
      crouchCoil.rHipZ = -0.12;
      crouchCoil.rKneeX = 0.56;
      crouchCoil.rFootX = -0.28;
      crouchCoil.rShoulderX = -1.18;
      crouchCoil.rShoulderZ = -0.94;
      crouchCoil.rElbowX = -0.44;
      crouchCoil.rHandY = -0.32;
      crouchCoil.rWeaponX = -1.22;
      crouchCoil.lShoulderX = -1.14;
      crouchCoil.lShoulderZ = 0.9;
      crouchCoil.lElbowX = -0.46;
      crouchCoil.lHandY = 0.32;
      crouchCoil.lWeaponX = -1.22;
      crouchCoil.capeLift = 0.38;

      const lungePeak = createZeroCustomPose();
      lungePeak.bodyBob = -0.01;
      lungePeak.bodyPitch = 0.26;
      lungePeak.torsoPitch = 0.14;
      lungePeak.torsoYaw = 0.08;
      lungePeak.headPitch = -0.18;
      lungePeak.lHipX = -0.66;
      lungePeak.lHipZ = 0.11;
      lungePeak.lKneeX = 0.9;
      lungePeak.lFootX = -0.34;
      lungePeak.rHipX = 0.54;
      lungePeak.rHipZ = -0.11;
      lungePeak.rKneeX = 0.2;
      lungePeak.rFootX = -0.36;
      lungePeak.rShoulderX = -1.34;
      lungePeak.rShoulderY = 0.34;
      lungePeak.rShoulderZ = 0.72;
      lungePeak.rElbowX = -0.28;
      lungePeak.rHandY = 0.48;
      lungePeak.rWeaponX = -1.54;
      lungePeak.lShoulderX = -1.28;
      lungePeak.lShoulderY = -0.3;
      lungePeak.lShoulderZ = -0.66;
      lungePeak.lElbowX = -0.32;
      lungePeak.lHandY = -0.48;
      lungePeak.lWeaponX = -1.5;
      lungePeak.capeLift = 0.98;

      const followSweep = { ...lungePeak };
      followSweep.bodyBob = -0.065;
      followSweep.bodyPitch = 0.28;
      followSweep.torsoPitch = 0.2;
      followSweep.torsoYaw = 0.2;
      followSweep.lKneeX = 0.96;
      followSweep.rShoulderX = -0.88;
      followSweep.rShoulderZ = -0.52;
      followSweep.rHandY = 0.15;
      followSweep.lShoulderX = -0.88;
      followSweep.lShoulderZ = 0.52;
      followSweep.lHandY = -0.15;
      followSweep.capeLift = 0.74;

      const elasticSnap = { ...ready };
      elasticSnap.bodyBob = -0.05;
      elasticSnap.torsoPitch = 0.1;
      elasticSnap.lKneeX = 0.56;
      elasticSnap.rKneeX = 0.46;

      return blendMultiStageMotion(
        [
          { t: 0.0, pose: ready },
          { t: 0.22, pose: crouchCoil, ease: 'smooth' },
          { t: 0.46, pose: lungePeak, ease: 'quart' },
          { t: 0.68, pose: followSweep, ease: 'smooth' },
          { t: 0.85, pose: elasticSnap, ease: 'cubic' },
          { t: 1.0, pose: ready, ease: 'quad' },
        ],
        p,
      );
    }

    // 1. Hip-Led Lightning Diagonal Reverse-Grip Cross-Slash + Foot Pivot
    const hipCoil = createZeroCustomPose();
    hipCoil.bodyYaw = 0.26;
    hipCoil.bodyRoll = -0.09;
    hipCoil.bodyBob = -0.05;
    hipCoil.torsoYaw = -0.74;
    hipCoil.torsoPitch = 0.14;
    hipCoil.torsoRoll = -0.08;
    hipCoil.headYaw = 0.68;
    hipCoil.headPitch = -0.08;
    hipCoil.lHipX = -0.32;
    hipCoil.lHipZ = 0.1;
    hipCoil.lKneeX = 0.54;
    hipCoil.lFootX = -0.22;
    hipCoil.rHipX = 0.28;
    hipCoil.rHipZ = -0.1;
    hipCoil.rKneeX = 0.52;
    hipCoil.rFootX = -0.24;
    hipCoil.rShoulderX = 0.56;
    hipCoil.rShoulderZ = -0.58;
    hipCoil.rElbowX = -1.08;
    hipCoil.rHandY = -0.4;
    hipCoil.rWeaponX = -1.24;
    hipCoil.lShoulderX = -1.08;
    hipCoil.lShoulderZ = 0.34;
    hipCoil.lElbowX = -1.14;
    hipCoil.capeLift = 0.36;

    const whipMid = createZeroCustomPose();
    whipMid.bodyYaw = -0.12;
    whipMid.bodyPitch = 0.22;
    whipMid.bodyRoll = 0.06;
    whipMid.bodyBob = -0.015;
    whipMid.torsoYaw = 0.32;
    whipMid.torsoPitch = 0.12;
    whipMid.headYaw = -0.28;
    whipMid.lHipX = -0.52;
    whipMid.lHipZ = 0.1;
    whipMid.lKneeX = 0.78;
    whipMid.rHipX = 0.44;
    whipMid.rHipZ = -0.1;
    whipMid.rKneeX = 0.26;
    whipMid.rShoulderX = -1.16;
    whipMid.rShoulderZ = -0.22;
    whipMid.rElbowX = -0.45;
    whipMid.rHandY = 0.25;
    whipMid.rWeaponX = -1.42;
    whipMid.lShoulderX = 0.28;
    whipMid.lShoulderZ = 0.36;
    whipMid.lElbowX = -0.68;
    whipMid.capeLift = 0.74;

    const strikeEnd = createZeroCustomPose();
    strikeEnd.bodyYaw = -0.28;
    strikeEnd.bodyPitch = 0.25;
    strikeEnd.bodyRoll = 0.1;
    strikeEnd.bodyBob = -0.03;
    strikeEnd.torsoYaw = 0.78;
    strikeEnd.torsoPitch = 0.16;
    strikeEnd.torsoRoll = 0.1;
    strikeEnd.headYaw = -0.64;
    strikeEnd.headPitch = -0.14;
    strikeEnd.lHipX = -0.6;
    strikeEnd.lHipZ = 0.1;
    strikeEnd.lKneeX = 0.86;
    strikeEnd.lFootX = -0.32;
    strikeEnd.rHipX = 0.5;
    strikeEnd.rHipZ = -0.1;
    strikeEnd.rKneeX = 0.2;
    strikeEnd.rFootX = -0.34;
    strikeEnd.rShoulderX = -1.52;
    strikeEnd.rShoulderY = 0.28;
    strikeEnd.rShoulderZ = 0.64;
    strikeEnd.rElbowX = -0.16;
    strikeEnd.rHandY = 0.52;
    strikeEnd.rWeaponX = -1.62;
    strikeEnd.lShoulderX = 0.62;
    strikeEnd.lShoulderZ = 0.4;
    strikeEnd.lElbowX = -0.46;
    strikeEnd.capeLift = 0.98;

    const followDecel = { ...strikeEnd };
    followDecel.bodyYaw = -0.32;
    followDecel.bodyBob = -0.06;
    followDecel.torsoYaw = 0.86;
    followDecel.headYaw = -0.68;
    followDecel.lKneeX = 0.94;
    followDecel.rShoulderX = -1.24;
    followDecel.rShoulderZ = 0.82;
    followDecel.capeLift = 0.76;

    return blendMultiStageMotion(
      [
        { t: 0.0, pose: ready },
        { t: 0.2, pose: hipCoil, ease: 'smooth' },
        { t: 0.34, pose: whipMid, ease: 'cubic' },
        { t: 0.48, pose: strikeEnd, ease: 'quart' },
        { t: 0.68, pose: followDecel, ease: 'smooth' },
        { t: 1.0, pose: ready, ease: 'quad' },
      ],
      p,
    );
  }

  // Signature Ultra-Realistic Heavy Warhammer Choreography for Xenon (true mass inertia, hip-first whip, impact knee-buckle & hoist)
  if (isHeavy) {
    if (isHammerSpecial) {
      // "Удар Ксенона": Deep squat load -> soaring leap (hammer lags behind body) -> brief overhead apex lock ->
      // violent spine crush impact -> knee-buckle ground rebound -> laborious heavy hoist back to poise
      const coil = createZeroCustomPose();
      coil.bodyYaw = 0.2;
      coil.bodyBob = -0.15;
      coil.bodyPitch = 0.2;
      coil.torsoPitch = 0.22;
      coil.torsoYaw = -0.26;
      coil.headYaw = 0.22;
      coil.headPitch = -0.14;
      coil.lHipX = -0.52;
      coil.lHipZ = 0.12;
      coil.lKneeX = 0.92;
      coil.lFootX = -0.26;
      coil.rHipX = 0.42;
      coil.rHipZ = -0.12;
      coil.rKneeX = 0.76;
      coil.rFootX = -0.28;
      coil.rShoulderX = 0.62;
      coil.rShoulderZ = -0.38;
      coil.rElbowX = -0.44;
      coil.rHandX = 0.26;
      coil.rHandZ = -0.16;
      coil.lShoulderX = -0.84;
      coil.lShoulderZ = 0.28;
      coil.lElbowX = -0.84;
      coil.capeLift = 0.56;

      const apexRise = createZeroCustomPose();
      apexRise.bodyYaw = -0.04;
      apexRise.bodyBob = 0.18;
      apexRise.bodyPitch = -0.12;
      apexRise.torsoPitch = -0.22;
      apexRise.torsoYaw = -0.18;
      apexRise.headYaw = 0.14;
      apexRise.headPitch = 0.12;
      apexRise.lHipX = -0.76;
      apexRise.lKneeX = 1.14;
      apexRise.rHipX = -0.24;
      apexRise.rKneeX = 0.72;
      apexRise.rShoulderX = -1.96; // weapon trails torso climb!
      apexRise.rShoulderZ = -0.18;
      apexRise.rElbowX = -0.42;
      apexRise.rHandX = 0.14;
      apexRise.lShoulderX = -1.78;
      apexRise.lShoulderZ = 0.26;
      apexRise.lElbowX = -0.48;
      apexRise.capeLift = 1.3;

      const apexHang = createZeroCustomPose();
      apexHang.bodyYaw = -0.1;
      apexHang.bodyBob = 0.22;
      apexHang.bodyPitch = -0.2;
      apexHang.torsoPitch = -0.32;
      apexHang.torsoYaw = -0.12;
      apexHang.headYaw = 0.1;
      apexHang.headPitch = 0.16;
      apexHang.lHipX = -0.78;
      apexHang.lKneeX = 1.16;
      apexHang.rHipX = -0.28;
      apexHang.rKneeX = 0.76;
      apexHang.rShoulderX = -2.62;
      apexHang.rShoulderZ = -0.1;
      apexHang.rElbowX = -0.24;
      apexHang.rHandX = 0.28;
      apexHang.lShoulderX = -2.28;
      apexHang.lShoulderZ = 0.24;
      apexHang.lElbowX = -0.34;
      apexHang.capeLift = 1.42;

      const slamImpact = createZeroCustomPose();
      slamImpact.bodyYaw = 0.08;
      slamImpact.bodyBob = -0.17;
      slamImpact.bodyPitch = 0.3;
      slamImpact.torsoPitch = 0.32;
      slamImpact.torsoYaw = 0.18;
      slamImpact.headYaw = -0.14;
      slamImpact.headPitch = -0.2;
      slamImpact.lHipX = -0.7;
      slamImpact.lHipZ = 0.12;
      slamImpact.lKneeX = 0.98;
      slamImpact.lFootX = -0.26;
      slamImpact.rHipX = 0.52;
      slamImpact.rHipZ = -0.12;
      slamImpact.rKneeX = 0.22;
      slamImpact.rFootX = -0.28;
      slamImpact.rShoulderX = -1.04;
      slamImpact.rShoulderZ = 0.04;
      slamImpact.rElbowX = -0.08;
      slamImpact.rHandX = 0.96;
      slamImpact.lShoulderX = 0.44;
      slamImpact.lShoulderZ = 0.4;
      slamImpact.lElbowX = -0.44;
      slamImpact.capeLift = 1.52;

      const reboundHold = { ...slamImpact };
      reboundHold.bodyBob = -0.19;
      reboundHold.torsoPitch = 0.36;
      reboundHold.lKneeX = 1.04;
      reboundHold.rShoulderX = -0.94;
      reboundHold.rHandX = 1.02;
      reboundHold.capeLift = 1.08;

      const hoistRecover = { ...ready };
      hoistRecover.bodyBob = -0.09;
      hoistRecover.bodyPitch = 0.14;
      hoistRecover.torsoPitch = 0.18;
      hoistRecover.rShoulderX = -0.76;
      hoistRecover.rElbowX = -0.92;
      hoistRecover.rHandX = 0.52;
      hoistRecover.lKneeX = 0.54;
      hoistRecover.rKneeX = 0.48;

      return blendMultiStageMotion(
        [
          { t: 0.0, pose: ready },
          { t: 0.18, pose: coil, ease: 'smooth' },
          { t: 0.38, pose: apexRise, ease: 'cubic' },
          { t: 0.47, pose: apexHang, ease: 'smooth' },
          { t: 0.58, pose: slamImpact, ease: 'quart' },
          { t: 0.76, pose: reboundHold, ease: 'smooth' },
          { t: 0.89, pose: hoistRecover, ease: 'cubic' },
          { t: 1.0, pose: ready, ease: 'quad' },
        ],
        p,
      );
    }

    if (strikeVariant === 3) {
      // 3. Titan Overhead Crushing Slam: weight-shift back -> full overhead hang -> spine-driven drop -> shock knee-buckle -> heavy hoist
      const windup = createZeroCustomPose();
      windup.bodyBob = 0.06;
      windup.bodyPitch = -0.14;
      windup.torsoPitch = -0.26;
      windup.torsoYaw = -0.16;
      windup.headYaw = 0.14;
      windup.headPitch = 0.14;
      windup.lHipX = -0.26;
      windup.lHipZ = 0.1;
      windup.lKneeX = 0.46;
      windup.rHipX = 0.2;
      windup.rHipZ = -0.1;
      windup.rKneeX = 0.36;
      windup.rShoulderX = -2.44;
      windup.rShoulderZ = -0.12;
      windup.rElbowX = -0.34;
      windup.rHandX = 0.26;
      windup.lShoulderX = -1.92;
      windup.lShoulderZ = 0.32;
      windup.lElbowX = -0.5;
      windup.capeLift = 0.58;

      const dropLag = createZeroCustomPose();
      dropLag.bodyPitch = 0.1;
      dropLag.bodyBob = -0.02;
      dropLag.torsoPitch = 0.08;
      dropLag.lHipX = -0.46;
      dropLag.lKneeX = 0.68;
      dropLag.rHipX = 0.34;
      dropLag.rKneeX = 0.28;
      dropLag.rShoulderX = -1.78; // head follows slightly behind spine acceleration
      dropLag.rElbowX = -0.18;
      dropLag.rHandX = 0.42;
      dropLag.lShoulderX = -0.82;
      dropLag.capeLift = 0.95;

      const strike = createZeroCustomPose();
      strike.bodyPitch = 0.26;
      strike.bodyBob = -0.14;
      strike.torsoPitch = 0.28;
      strike.torsoYaw = 0.16;
      strike.headYaw = -0.14;
      strike.headPitch = -0.18;
      strike.lHipX = -0.64;
      strike.lHipZ = 0.1;
      strike.lKneeX = 0.92;
      strike.lFootX = -0.24;
      strike.rHipX = 0.5;
      strike.rHipZ = -0.1;
      strike.rKneeX = 0.2;
      strike.rFootX = -0.26;
      strike.rShoulderX = -1.04;
      strike.rShoulderZ = 0.04;
      strike.rElbowX = -0.1;
      strike.rHandX = 0.9;
      strike.lShoulderX = 0.38;
      strike.lShoulderZ = 0.36;
      strike.lElbowX = -0.48;
      strike.capeLift = 1.24;

      const follow = { ...strike };
      follow.bodyBob = -0.16;
      follow.torsoPitch = 0.33;
      follow.lKneeX = 0.98;
      follow.rShoulderX = -0.94;
      follow.rHandX = 0.96;
      follow.capeLift = 0.9;

      const hoist = { ...ready };
      hoist.bodyBob = -0.08;
      hoist.torsoPitch = 0.12;
      hoist.rShoulderX = -0.78;
      hoist.rElbowX = -0.88;
      hoist.rHandX = 0.46;

      return blendMultiStageMotion(
        [
          { t: 0.0, pose: ready },
          { t: 0.28, pose: windup, ease: 'smooth' },
          { t: 0.42, pose: dropLag, ease: 'cubic' },
          { t: 0.56, pose: strike, ease: 'quart' },
          { t: 0.74, pose: follow, ease: 'smooth' },
          { t: 0.88, pose: hoist, ease: 'cubic' },
          { t: 1.0, pose: ready, ease: 'quad' },
        ],
        p,
      );
    }

    if (strikeVariant === 2) {
      // 2. Heavy Left-to-Right Backhand Warhammer Smash: cross-torso coil -> hip-whip lag -> sweeping horizontal crush -> deceleration brace
      const windup = createZeroCustomPose();
      windup.bodyYaw = -0.28;
      windup.bodyRoll = -0.07;
      windup.bodyBob = -0.09;
      windup.torsoYaw = 0.74;
      windup.torsoPitch = 0.1;
      windup.headYaw = -0.62;
      windup.lHipX = -0.34;
      windup.lHipZ = 0.1;
      windup.lKneeX = 0.6;
      windup.rHipX = 0.3;
      windup.rHipZ = -0.1;
      windup.rKneeX = 0.46;
      windup.rShoulderX = -0.94;
      windup.rShoulderZ = 0.58;
      windup.rElbowX = -0.76;
      windup.rHandX = 0.36;
      windup.rHandZ = 0.24;
      windup.lShoulderX = 0.34;
      windup.lShoulderZ = 0.34;
      windup.lElbowX = -0.6;
      windup.capeLift = 0.38;

      const hipLead = createZeroCustomPose();
      hipLead.bodyYaw = 0.06;
      hipLead.bodyPitch = 0.1;
      hipLead.bodyBob = -0.05;
      hipLead.torsoYaw = 0.12; // hips snapped ahead while hammer lags!
      hipLead.headYaw = -0.08;
      hipLead.lHipX = -0.44;
      hipLead.lKneeX = 0.66;
      hipLead.rHipX = 0.36;
      hipLead.rKneeX = 0.34;
      hipLead.rShoulderX = -1.2;
      hipLead.rShoulderZ = 0.08;
      hipLead.rElbowX = -0.36;
      hipLead.rHandX = 0.52;
      hipLead.lShoulderX = -0.3;
      hipLead.lShoulderZ = 0.36;
      hipLead.capeLift = 0.68;

      const strike = createZeroCustomPose();
      strike.bodyYaw = 0.28;
      strike.bodyPitch = 0.16;
      strike.bodyRoll = 0.09;
      strike.bodyBob = -0.04;
      strike.torsoYaw = -0.74;
      strike.torsoPitch = -0.04;
      strike.headYaw = 0.62;
      strike.lHipX = -0.54;
      strike.lHipZ = 0.1;
      strike.lKneeX = 0.78;
      strike.lFootX = -0.22;
      strike.rHipX = 0.44;
      strike.rHipZ = -0.1;
      strike.rKneeX = 0.24;
      strike.rFootX = -0.24;
      strike.rShoulderX = -1.5;
      strike.rShoulderZ = -0.74;
      strike.rElbowX = -0.12;
      strike.rHandX = 0.72;
      strike.rHandZ = 0.22;
      strike.lShoulderX = -0.66;
      strike.lShoulderZ = 0.38;
      strike.lElbowX = -0.68;
      strike.capeLift = 1.08;

      const follow = { ...strike };
      follow.bodyYaw = 0.34;
      follow.bodyBob = -0.08;
      follow.torsoYaw = -0.84;
      follow.headYaw = 0.68;
      follow.rShoulderX = -1.66;
      follow.rShoulderZ = -0.86;
      follow.capeLift = 0.84;

      return blendMultiStageMotion(
        [
          { t: 0.0, pose: ready },
          { t: 0.24, pose: windup, ease: 'smooth' },
          { t: 0.38, pose: hipLead, ease: 'cubic' },
          { t: 0.52, pose: strike, ease: 'quart' },
          { t: 0.7, pose: follow, ease: 'smooth' },
          { t: 1.0, pose: ready, ease: 'quad' },
        ],
        p,
      );
    }

    // 1. Wide Right-to-Left Heavy Diagonal Forehand Cleave with True Hip-First Inertia & Heavy Hoist
    const windup = createZeroCustomPose();
    windup.bodyYaw = 0.26;
    windup.bodyPitch = -0.07;
    windup.bodyRoll = -0.07;
    windup.bodyBob = -0.06;
    windup.torsoYaw = -0.78;
    windup.torsoPitch = -0.08;
    windup.headYaw = 0.64;
    windup.lHipX = -0.28;
    windup.lHipZ = 0.1;
    windup.lKneeX = 0.5;
    windup.rHipX = 0.28;
    windup.rHipZ = -0.1;
    windup.rKneeX = 0.46;
    windup.rShoulderX = -1.16;
    windup.rShoulderZ = -0.78;
    windup.rElbowX = -0.62;
    windup.rHandX = 0.24;
    windup.rHandZ = -0.2;
    windup.lShoulderX = -0.9;
    windup.lShoulderZ = 0.3;
    windup.lElbowX = -0.74;
    windup.capeLift = 0.36;

    const hipLead = createZeroCustomPose();
    hipLead.bodyYaw = -0.04;
    hipLead.bodyPitch = 0.12;
    hipLead.bodyBob = -0.04;
    hipLead.torsoYaw = -0.12; // hips lead rotation before hammer whips through
    hipLead.headYaw = 0.1;
    hipLead.lHipX = -0.46;
    hipLead.lKneeX = 0.7;
    hipLead.rHipX = 0.38;
    hipLead.rKneeX = 0.32;
    hipLead.rShoulderX = -1.18;
    hipLead.rShoulderZ = -0.14;
    hipLead.rElbowX = -0.34;
    hipLead.rHandX = 0.48;
    hipLead.lShoulderX = -0.32;
    hipLead.lShoulderZ = 0.34;
    hipLead.capeLift = 0.7;

    const strike = createZeroCustomPose();
    strike.bodyYaw = -0.26;
    strike.bodyPitch = 0.2;
    strike.bodyRoll = 0.08;
    strike.bodyBob = -0.1;
    strike.torsoYaw = 0.76;
    strike.torsoPitch = 0.16;
    strike.headYaw = -0.62;
    strike.headPitch = -0.1;
    strike.lHipX = -0.6;
    strike.lHipZ = 0.1;
    strike.lKneeX = 0.88;
    strike.lFootX = -0.24;
    strike.rHipX = 0.46;
    strike.rHipZ = -0.1;
    strike.rKneeX = 0.22;
    strike.rFootX = -0.26;
    strike.rShoulderX = -1.2;
    strike.rShoulderZ = 0.56;
    strike.rElbowX = -0.12;
    strike.rHandX = 0.78;
    strike.rHandZ = -0.24;
    strike.lShoulderX = 0.44;
    strike.lShoulderZ = 0.36;
    strike.lElbowX = -0.48;
    strike.capeLift = 1.06;

    const follow = { ...strike };
    follow.bodyYaw = -0.32;
    follow.bodyBob = -0.12;
    follow.torsoYaw = 0.86;
    follow.headYaw = -0.68;
    follow.lKneeX = 0.94;
    follow.rShoulderZ = 0.7;
    follow.rHandX = 0.88;
    follow.capeLift = 0.82;

    return blendMultiStageMotion(
      [
        { t: 0.0, pose: ready },
        { t: 0.26, pose: windup, ease: 'smooth' },
        { t: 0.4, pose: hipLead, ease: 'cubic' },
        { t: 0.54, pose: strike, ease: 'quart' },
        { t: 0.72, pose: follow, ease: 'smooth' },
        { t: 1.0, pose: ready, ease: 'quad' },
      ],
      p,
    );
  }

  if (strikeVariant === 3) {
    const windup = createZeroCustomPose();
    windup.bodyYaw = 0.14;
    windup.bodyPitch = -0.14;
    windup.bodyBob = -0.04;
    windup.torsoPitch = -0.3;
    windup.torsoYaw = -0.22;
    windup.headYaw = 0.2;
    windup.lHipX = -0.26;
    windup.lKneeX = 0.46;
    windup.lFootX = -0.18;
    windup.rHipX = 0.24;
    windup.rKneeX = 0.36;
    windup.rFootX = -0.2;
    windup.rShoulderX = -2.35;
    windup.rShoulderZ = -0.18;
    windup.lShoulderX = -1.24;
    windup.lShoulderZ = 0.24;
    windup.rElbowX = -0.5;
    windup.capeLift = 0.3;

    const strike = createZeroCustomPose();
    strike.bodyYaw = -0.14;
    strike.bodyPitch = 0.26;
    strike.bodyRoll = 0.05;
    strike.bodyBob = -0.1;
    strike.torsoPitch = 0.28;
    strike.torsoYaw = 0.32;
    strike.headYaw = -0.26;
    strike.lHipX = -0.62;
    strike.lHipZ = 0.08;
    strike.lKneeX = 0.88;
    strike.lFootX = -0.24;
    strike.rHipX = 0.5;
    strike.rHipZ = -0.08;
    strike.rKneeX = 0.18;
    strike.rFootX = -0.26;
    strike.rShoulderX = -0.32;
    strike.lShoulderX = -0.45;
    strike.rElbowX = -0.12;
    strike.rHandX = 0.32;
    strike.capeLift = 0.98;

    const follow = { ...strike };
    follow.bodyBob = -0.13;
    follow.torsoPitch = 0.35;
    follow.lKneeX = 0.94;
    follow.rShoulderX = -0.22;
    follow.capeLift = 0.8;

    return blendFourStageMotion(ready, windup, strike, follow, p, 0.26, 0.52, 0.69);
  }

  if (strikeVariant === 2) {
    const windup = createZeroCustomPose();
    windup.bodyYaw = 0.24;
    windup.bodyPitch = -0.05;
    windup.bodyRoll = -0.07;
    windup.bodyBob = -0.03;
    windup.torsoYaw = -0.72;
    windup.headYaw = 0.62;
    windup.lHipX = -0.22;
    windup.lKneeX = 0.44;
    windup.lFootX = -0.18;
    windup.rHipX = 0.28;
    windup.rKneeX = 0.38;
    windup.rFootX = -0.22;
    windup.rShoulderX = -1.12;
    windup.rShoulderZ = -0.84;
    windup.rElbowX = -0.28;
    windup.lShoulderX = -0.7;
    windup.lShoulderZ = 0.26;
    windup.capeLift = 0.24;

    const strike = createZeroCustomPose();
    strike.bodyYaw = -0.22;
    strike.bodyPitch = 0.18;
    strike.bodyRoll = 0.08;
    strike.bodyBob = -0.08;
    strike.torsoYaw = 0.72;
    strike.headYaw = -0.6;
    strike.rShoulderX = -1.1;
    strike.rShoulderZ = 0.84;
    strike.rElbowX = -0.22;
    strike.rHandZ = -0.28;
    strike.lShoulderX = -0.58;
    strike.lShoulderZ = 0.32;
    strike.lHipX = -0.52;
    strike.lHipZ = 0.09;
    strike.lKneeX = 0.74;
    strike.lFootX = -0.22;
    strike.rHipX = 0.4;
    strike.rHipZ = -0.09;
    strike.rKneeX = 0.22;
    strike.rFootX = -0.22;
    strike.capeLift = 0.86;

    const follow = { ...strike };
    follow.bodyYaw = -0.28;
    follow.torsoYaw = 0.82;
    follow.rShoulderZ = 0.96;
    follow.capeLift = 0.68;

    return blendFourStageMotion(ready, windup, strike, follow, p, 0.24, 0.50, 0.67);
  }

  const windup = createZeroCustomPose();
  windup.bodyYaw = 0.2;
  windup.bodyPitch = -0.1;
  windup.bodyRoll = -0.06;
  windup.bodyBob = -0.03;
  windup.torsoYaw = -0.58;
  windup.headYaw = 0.5;
  windup.lHipX = -0.22;
  windup.lKneeX = 0.42;
  windup.lFootX = -0.16;
  windup.rHipX = 0.28;
  windup.rKneeX = 0.36;
  windup.rFootX = -0.2;
  windup.rShoulderX = 0.58;
  windup.rElbowX = -0.92;
  windup.lShoulderX = -0.78;
  windup.lShoulderZ = 0.28;
  windup.capeLift = 0.22;

  const strike = createZeroCustomPose();
  strike.bodyYaw = -0.2;
  strike.bodyPitch = 0.24;
  strike.bodyRoll = 0.06;
  strike.bodyBob = -0.08;
  strike.torsoYaw = 0.56;
  strike.headYaw = -0.48;
  strike.lHipX = -0.56;
  strike.lHipZ = 0.08;
  strike.lKneeX = 0.78;
  strike.lFootX = -0.22;
  strike.rHipX = 0.46;
  strike.rHipZ = -0.08;
  strike.rKneeX = 0.18;
  strike.rFootX = -0.24;
  strike.rShoulderX = -1.48;
  strike.rElbowX = -0.08;
  strike.rHandX = 0.35;
  strike.lShoulderX = -0.66;
  strike.lShoulderZ = 0.28;
  strike.capeLift = 0.86;

  const follow = { ...strike };
  follow.bodyBob = -0.11;
  follow.bodyYaw = -0.24;
  follow.torsoYaw = 0.64;
  follow.lKneeX = 0.86;
  follow.rShoulderX = -1.36;
  follow.capeLift = 0.68;

  return blendFourStageMotion(ready, windup, strike, follow, p, 0.24, 0.50, 0.66);
}

export function buildCustomJumpPose(params: {
  mode: 'jump' | 'air_strike';
  velY?: number;
  strikeProgress?: number;
  isHeavy?: boolean;
  isTwinBlade?: boolean;
  elapsed?: number;
  strafeRatio?: number;
}): CustomBonePose {
  const {
    mode,
    velY = 0,
    strikeProgress = 0,
    isHeavy = false,
    isTwinBlade = false,
    elapsed = 0,
    strafeRatio = 0,
  } = params;
  const rise = THREE.MathUtils.clamp(velY / 7.5, -1, 1);
  const launchLift = Math.max(0, rise);
  const fallBrace = Math.max(0, -rise);

  const jump = createZeroCustomPose();
  if (isTwinBlade) {
    jump.torsoPitch = 0.14 - launchLift * 0.08 + fallBrace * 0.12;
    jump.headPitch = rise < -0.2 ? 0.1 : -0.1;
    jump.lShoulderX = 0.42 - launchLift * 0.52 + fallBrace * 0.25;
    jump.lShoulderZ = 0.46 + fallBrace * 0.18;
    jump.lElbowX = -0.68;
    jump.rShoulderX = -0.76 - launchLift * 0.28 + fallBrace * 0.2;
    jump.rShoulderZ = -0.32 - fallBrace * 0.16;
    jump.rElbowX = -1.02;
  } else if (isHeavy) {
    jump.torsoPitch = 0.08 - launchLift * 0.06 + fallBrace * 0.1;
    jump.torsoYaw = -0.16;
    jump.headYaw = 0.14;
    jump.headPitch = rise < -0.2 ? 0.12 : -0.04;
    jump.rShoulderX = -0.82 - launchLift * 0.32 + fallBrace * 0.18;
    jump.rShoulderZ = -0.26 - fallBrace * 0.12;
    jump.rElbowX = -0.78;
    jump.rHandX = 0.34;
    jump.rHandZ = -0.14;
    jump.lShoulderX = -0.54 - launchLift * 0.26 + fallBrace * 0.22;
    jump.lShoulderZ = 0.34 + fallBrace * 0.18;
    jump.lElbowX = -0.76;
  } else {
    jump.lShoulderX = -0.64 - launchLift * 0.28 + fallBrace * 0.22;
    jump.lShoulderZ = 0.28 + fallBrace * 0.18;
    jump.lElbowX = -0.58;
    jump.rShoulderX = -0.45 - launchLift * 0.26 + fallBrace * 0.18;
    jump.rShoulderZ = -0.24 - fallBrace * 0.16;
    jump.rElbowX = -0.65;
  }
  applyFluidAirborneLegs(jump, velY, elapsed, strafeRatio);

  if (mode === 'jump') {
    return jump;
  }

  const p = THREE.MathUtils.clamp(strikeProgress, 0, 1);
  if (isTwinBlade) {
    // Falcon Dive Twin-Kunai X-Slash!
    const windup = createZeroCustomPose();
    windup.bodyPitch = -0.2;
    windup.torsoPitch = -0.24;
    windup.lHipX = -0.88;
    windup.lKneeX = 1.32;
    windup.rHipX = -0.32;
    windup.rKneeX = 0.85;
    windup.lFootX = 0.42;
    windup.rFootX = 0.42;
    windup.rShoulderX = -1.85;
    windup.rShoulderZ = -0.78;
    windup.rElbowX = -0.45;
    windup.lShoulderX = -1.85;
    windup.lShoulderZ = 0.78;
    windup.lElbowX = -0.45;
    windup.rWeaponX = -1.2;
    windup.lWeaponX = -1.2;
    windup.capeLift = 0.8;

    const plunge = createZeroCustomPose();
    plunge.bodyPitch = 0.46;
    plunge.torsoPitch = 0.26;
    plunge.headPitch = -0.18;
    plunge.lHipX = -0.45;
    plunge.lKneeX = 0.48;
    plunge.rHipX = 0.58;
    plunge.rKneeX = 0.56;
    plunge.rFootX = 0.36;
    plunge.rShoulderX = -1.32;
    plunge.rShoulderZ = 0.62;
    plunge.rElbowX = -0.35;
    plunge.lShoulderX = -1.32;
    plunge.lShoulderZ = -0.62;
    plunge.lElbowX = -0.35;
    plunge.rWeaponX = -1.5;
    plunge.lWeaponX = -1.5;
    plunge.capeLift = 1.2;

    const follow = { ...plunge };
    follow.bodyPitch = 0.52;
    follow.torsoPitch = 0.34;
    follow.lHipX = -0.32;
    follow.lKneeX = 0.64;
    follow.rHipX = 0.44;
    follow.rKneeX = 0.74;
    follow.rShoulderX = -1.05;
    follow.lShoulderX = -1.05;
    follow.capeLift = 1.32;

    return blendFourStageMotion(jump, windup, plunge, follow, p, 0.24, 0.52, 0.7);
  }

  const windup = createZeroCustomPose();
  windup.bodyPitch = -0.16;
  windup.torsoPitch = -0.24;
  windup.torsoYaw = isHeavy ? -0.18 : -0.35;
  windup.headYaw = isHeavy ? 0.14 : 0.28;
  windup.lHipX = -0.84;
  windup.lKneeX = 1.24;
  windup.rHipX = -0.24;
  windup.rKneeX = 0.72;
  windup.rShoulderX = isHeavy ? -2.38 : -2.35;
  windup.rShoulderZ = isHeavy ? -0.12 : 0;
  windup.rElbowX = isHeavy ? -0.36 : -0.52;
  windup.rHandX = isHeavy ? 0.22 : 0;
  windup.lShoulderX = isHeavy ? -1.85 : -0.92;
  windup.lShoulderZ = isHeavy ? 0.28 : 0;
  windup.lElbowX = -0.52;
  windup.capeLift = 0.65;

  const plunge = createZeroCustomPose();
  plunge.bodyPitch = isHeavy ? 0.32 : 0.42;
  plunge.torsoPitch = isHeavy ? 0.26 : 0.32;
  plunge.torsoYaw = isHeavy ? 0.16 : 0.32;
  plunge.headYaw = isHeavy ? -0.14 : -0.24;
  plunge.lHipX = -0.42;
  plunge.lKneeX = 0.44;
  plunge.rHipX = 0.56;
  plunge.rKneeX = 0.6;
  plunge.rShoulderX = isHeavy ? -1.06 : -0.42;
  plunge.rShoulderZ = isHeavy ? 0.04 : 0;
  plunge.rElbowX = -0.12;
  plunge.rHandX = isHeavy ? 0.88 : 0;
  plunge.lShoulderX = isHeavy ? 0.36 : -0.48;
  plunge.lShoulderZ = isHeavy ? 0.34 : 0.35;
  plunge.lElbowX = isHeavy ? -0.48 : 0;
  plunge.capeLift = 1.24;

  const follow = { ...plunge };
  follow.bodyPitch = isHeavy ? 0.38 : 0.48;
  follow.torsoPitch = isHeavy ? 0.32 : 0.38;
  follow.torsoYaw = isHeavy ? 0.2 : 0.4;
  follow.lKneeX = 0.56;
  follow.rKneeX = 0.74;
  follow.rShoulderX = isHeavy ? -0.98 : -0.28;
  follow.rHandX = isHeavy ? 0.94 : 0;
  follow.capeLift = 1.32;

  return blendFourStageMotion(jump, windup, plunge, follow, p, 0.25, 0.54, 0.72);
}

const BONE_LAMBDA_MULT: Record<keyof CustomBonePose, number> = {
  bodyYaw: 0.9,
  bodyPitch: 0.92,
  bodyRoll: 0.88,
  bodyBob: 0.95,
  torsoYaw: 0.96,
  torsoPitch: 0.96,
  torsoRoll: 0.92,
  headYaw: 0.76,
  headPitch: 0.76,
  lHipX: 1.05,
  lHipZ: 1.0,
  rHipX: 1.05,
  rHipZ: 1.0,
  lKneeX: 1.08,
  rKneeX: 1.08,
  lFootX: 1.12,
  rFootX: 1.12,
  lShoulderX: 1.14,
  lShoulderY: 1.12,
  lShoulderZ: 1.12,
  lElbowX: 1.24,
  lHandX: 1.3,
  lHandY: 1.25,
  lHandZ: 1.25,
  lWeaponX: 1.32,
  rShoulderX: 1.14,
  rShoulderY: 1.12,
  rShoulderZ: 1.12,
  rElbowX: 1.24,
  rHandX: 1.3,
  rHandY: 1.25,
  rHandZ: 1.25,
  rWeaponX: 1.32,
  tailPitch: 0.66,
  tailYaw: 0.64,
  capeLift: 0.6,
};

export function applyCustomBonePose(
  rig: CustomLbxRig,
  target: CustomBonePose,
  dt: number,
  lambda = 18,
  freeAimHead = false,
) {
  if (!rig.poseState) {
    rig.poseState = createZeroCustomPose();
  }
  rig.animTime = (rig.animTime ?? 0) + dt;
  const s = rig.poseState;
  while (s.bodyYaw - target.bodyYaw > Math.PI) {
    s.bodyYaw -= Math.PI * 2;
  }
  while (s.bodyYaw - target.bodyYaw < -Math.PI) {
    s.bodyYaw += Math.PI * 2;
  }
  const keys = Object.keys(target) as (keyof CustomBonePose)[];
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    s[k] = THREE.MathUtils.damp(s[k], target[k], lambda * (BONE_LAMBDA_MULT[k] ?? 1), dt);
  }

  rig.bodyGroup.rotation.order = 'YXZ';
  rig.bodyGroup.rotation.y = s.bodyYaw;
  rig.bodyGroup.rotation.x = s.bodyPitch;
  rig.bodyGroup.rotation.z = s.bodyRoll;
  rig.bodyGroup.position.y = s.bodyBob;

  const b = rig.bones;
  if (!b || !rig.hasSkeleton) {
    // Fallback for unrigged static .obj / .stl meshes: twist, pitch & bank whole body
    rig.bodyGroup.rotation.y = s.bodyYaw + s.torsoYaw * 0.5;
    rig.bodyGroup.rotation.x = s.bodyPitch + s.torsoPitch * 0.7;
    rig.bodyGroup.rotation.z = s.bodyRoll;
    return;
  }

  // Spine & Head (single-spine rigs like Kunoichi get 100% on spineUpper)
  if (b.spineLower) {
    rotateBoneInCharSpace(b.spineLower, s.torsoPitch * 0.45, s.torsoYaw * 0.55, s.torsoRoll * 0.45);
    rotateBoneInCharSpace(b.spineUpper, s.torsoPitch * 0.55, s.torsoYaw * 0.45, s.torsoRoll * 0.55);
  } else {
    rotateBoneInCharSpace(b.spineUpper, s.torsoPitch, s.torsoYaw, s.torsoRoll);
  }

  const headPitch = freeAimHead
    ? s.headPitch
    : Math.abs(s.headPitch) > 0.001
      ? s.headPitch
      : -s.torsoPitch * 0.25;
  const headYaw = freeAimHead
    ? s.headYaw
    : Math.abs(s.headYaw) > 0.001
      ? s.headYaw
      : -s.torsoYaw * 0.35;
  if (b.neck) {
    rotateBoneInCharSpace(b.neck, headPitch * 0.35, headYaw * 0.35, 0);
    rotateBoneInCharSpace(b.head, headPitch * 0.65, headYaw * 0.65, 0);
  } else {
    rotateBoneInCharSpace(b.head, headPitch, headYaw, 0);
  }

  // Arms + Wrists + Built-in Hand Weapons
  rotateBoneInCharSpace(b.shoulderR, s.rShoulderX, s.rShoulderY, s.rShoulderZ);
  rotateBoneInCharSpace(b.elbowR, s.rElbowX, 0, 0);
  rotateBoneInCharSpace(b.handR, s.rHandX, s.rHandY, s.rHandZ);
  rotateBoneInCharSpace(b.weaponR, s.rWeaponX, 0, 0);

  rotateBoneInCharSpace(b.shoulderL, s.lShoulderX, s.lShoulderY, s.lShoulderZ);
  rotateBoneInCharSpace(b.elbowL, s.lElbowX, 0, 0);
  rotateBoneInCharSpace(b.handL, s.lHandX, s.lHandY, s.lHandZ);
  rotateBoneInCharSpace(b.weaponL, s.lWeaponX, 0, 0);

  // Legs + Ankles
  rotateBoneInCharSpace(b.hipL, s.lHipX, 0, s.lHipZ);
  rotateBoneInCharSpace(b.kneeL, s.lKneeX, 0, 0);
  rotateBoneInCharSpace(b.footL, s.lFootX, 0, 0);
  rotateBoneInCharSpace(b.hipR, s.rHipX, 0, s.rHipZ);
  rotateBoneInCharSpace(b.kneeR, s.rKneeX, 0, 0);
  rotateBoneInCharSpace(b.footR, s.rFootX, 0, 0);

  // Tail (e.g. Hunter wolf tail)
  if (b.tail) {
    rotateBoneInCharSpace(b.tail, s.tailPitch, s.tailYaw, 0);
  }

  // Cape segments billow backward with organic traveling cloth ripple
  const tWave = rig.animTime;
  const waveAmp = 0.025 + Math.min(1.2, Math.abs(s.capeLift)) * 0.055;
  const singleFullCape = b.capeL.length === 1 && b.capeR.length === 0;
  b.capeL.forEach((seg, idx) => {
    const pitchFactor = singleFullCape ? 0.52 : 0.35 + idx * 0.12;
    const flutter = Math.sin(tWave * 4.6 - idx * 0.85) * waveAmp;
    const rollFlare = singleFullCape ? flutter * 0.45 : 0.08 * s.capeLift + flutter * 0.4;
    rotateBoneInCharSpace(seg, s.capeLift * pitchFactor + flutter, 0, rollFlare);
  });
  b.capeR.forEach((seg, idx) => {
    const flutter = Math.sin(tWave * 4.6 - idx * 0.85 + 0.5) * waveAmp;
    rotateBoneInCharSpace(
      seg,
      s.capeLift * (0.35 + idx * 0.12) + flutter,
      0,
      -0.08 * s.capeLift - flutter * 0.4,
    );
  });
}

function isBuiltInWeaponNode(name: string): boolean {
  if (!name) return false;
  // GLTFLoader sanitizes node names via PropertyBinding.sanitizeNodeName,
  // stripping '.' (e.g. 'Cylinder.009' -> 'Cylinder009', 'Sphere.030' -> 'Sphere030').
  // Note: Plane001 is the mirrored right-hand palm (matches left hand 'Khớp cổ tay'), so keep it visible!
  const clean = name.replace(/\./g, '');
  if (
    /^Cylinder009($|_)/i.test(clean) ||
    /^Bone008(_end)?_/i.test(clean) ||
    /^Sphere030($|_)/i.test(clean)
  ) {
    return true;
  }
  return /(?:^|[^a-z])(weapon|lance|spear|sword|blade|shield|rifle|gun|hammer|halberd|axe|bow|staff|kunai|khien|giao|kiem)(?:[^a-z]|$)/i.test(
    name,
  );
}

function bindSkeletonBones(
  rawModel: THREE.Object3D,
  charFrame: THREE.Object3D,
): CustomBoneSet {
  const charWorldQuat = charFrame.getWorldQuaternion(new THREE.Quaternion());
  const charWorldQuatInv = charWorldQuat.clone().invert();

  const makeBound = (node: THREE.Object3D): BoundBone => {
    const parentQuat = node.parent
      ? node.parent.getWorldQuaternion(new THREE.Quaternion())
      : charWorldQuat.clone();
    return {
      node,
      restLocalQuat: node.quaternion.clone(),
      parentInCharQuat: charWorldQuatInv.clone().multiply(parentQuat),
    };
  };

  const set: CustomBoneSet = { capeL: [], capeR: [] };

  rawModel.traverse((node) => {
    const n = node.name || '';
    if (!n) return;
    const isBone = (node as THREE.Bone).isBone === true;

    if (!set.spineLower && /^(Bone\.?001|Spine1?|lower.?spine|waist)/i.test(n)) {
      set.spineLower = makeBound(node);
    } else if (!set.spineUpper && /^(Bone\.?002|Spine2|upper.?spine|upper.?chest|chest|torso(_\d+)?)/i.test(n)) {
      if (isBone || !/^TORSO$/i.test(n)) {
        set.spineUpper = makeBound(node);
      }
    } else if (!set.neck && isBone && /^NECK(_\d+)?$/i.test(n)) {
      set.neck = makeBound(node);
    } else if (!set.head && /^(Bone\.?003(?!_end)|Head(?!_end)(_\d+)?|neck)/i.test(n)) {
      if (isBone || !/^HEAD$/i.test(n)) {
        set.head = makeBound(node);
      }
    } else if (
      !set.shoulderL &&
      (/^(uper.?hand_L|upper.?arm_L|shoulder_L|LeftArm)/i.test(n) ||
        (isBone && /^L_(SHOULDER|UPPER.?ARM|ARM)(_\d+)?$/i.test(n)))
    ) {
      set.shoulderL = makeBound(node);
    } else if (
      !set.elbowL &&
      (/^(lower.?hand_L|forearm_L|elbow_L|LeftForeArm)/i.test(n) ||
        (isBone && /^L_(ELBOW|FORE.?ARM)(_\d+)?$/i.test(n)))
    ) {
      set.elbowL = makeBound(node);
    } else if (
      !set.handL &&
      (/^(hand_L|wrist_L|LeftHand)/i.test(n) ||
        (isBone && /^L_(BRUSH|HAND|WRIST)(_\d+)?$/i.test(n)))
    ) {
      set.handL = makeBound(node);
    } else if (!set.weaponL && isBone && /^L_LASER_KUNAI(_\d+)?$/i.test(n)) {
      set.weaponL = makeBound(node);
    } else if (
      !set.shoulderR &&
      (/^(uper.?hand_R|upper.?arm_R|shoulder_R|RightArm)/i.test(n) ||
        (isBone && /^R_(SHOULDER|UPPER.?ARM|ARM)(_\d+)?$/i.test(n)))
    ) {
      set.shoulderR = makeBound(node);
    } else if (
      !set.elbowR &&
      (/^(lower.?hand_R|forearm_R|elbow_R|RightForeArm)/i.test(n) ||
        (isBone && /^R_(ELBOW|FORE.?ARM)(_\d+)?$/i.test(n)))
    ) {
      set.elbowR = makeBound(node);
    } else if (
      !set.handR &&
      (/^(hand_R|wrist_R|RightHand)/i.test(n) ||
        (isBone && /^R_(BRUSH|HAND|WRIST)(_\d+)?$/i.test(n)))
    ) {
      set.handR = makeBound(node);
    } else if (!set.weaponR && isBone && /^R_LASER_KUNAI(_\d+)?$/i.test(n)) {
      set.weaponR = makeBound(node);
    } else if (
      !set.hipL &&
      (/^(uper.?leg_L|thigh_L|LeftUpLeg)/i.test(n) ||
        (isBone && /^L_(HIP|THIGH|UPPER.?LEG)(_\d+)?$/i.test(n)))
    ) {
      set.hipL = makeBound(node);
    } else if (
      !set.kneeL &&
      (/^(lower.?leg_L|shin_L|LeftLeg)/i.test(n) ||
        (isBone && /^L_(KNEE|CALF|SHIN|LOWER.?LEG)(_\d+)?$/i.test(n)))
    ) {
      set.kneeL = makeBound(node);
    } else if (!set.footL && isBone && /^L_FOOT(_\d+)?$/i.test(n)) {
      set.footL = makeBound(node);
    } else if (
      !set.hipR &&
      (/^(uper.?leg_R|thigh_R|RightUpLeg)/i.test(n) ||
        (isBone && /^R_(HIP|THIGH|UPPER.?LEG)(_\d+)?$/i.test(n)))
    ) {
      set.hipR = makeBound(node);
    } else if (
      !set.kneeR &&
      (/^(lower.?leg_R|shin_R|RightLeg)/i.test(n) ||
        (isBone && /^R_(KNEE|CALF|SHIN|LOWER.?LEG)(_\d+)?$/i.test(n)))
    ) {
      set.kneeR = makeBound(node);
    } else if (!set.footR && isBone && /^R_FOOT(_\d+)?$/i.test(n)) {
      set.footR = makeBound(node);
    } else if (/^k[1-4]_L(?!.*_end)/i.test(n)) {
      set.capeL.push(makeBound(node));
    } else if (/^k[1-4]_R(?!.*_end)/i.test(n)) {
      set.capeR.push(makeBound(node));
    }
  });

  return set;
}

interface ExtractedCluster {
  triIndices: number[];
  center: THREE.Vector3;
  box: THREE.Box3;
}

type PartTag = 'missile' | 'tail' | 'cape' | 'pelvis' | 'leg' | 'arm' | 'body' | 'head';

interface TriChunk {
  mesh: THREE.Mesh;
  toWrapper: THREE.Matrix4;
  tris: number[];
}

function getMeshPartTag(mesh: THREE.Mesh): PartTag | null {
  const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
  const parts: string[] = [mesh.name || '', (mat && mat.name) || ''];
  let cur: THREE.Object3D | null = mesh.parent;
  for (let depth = 0; cur && depth < 4; depth += 1) {
    if (cur.name) parts.push(cur.name);
    cur = cur.parent;
  }
  const label = parts.join(' ').toLowerCase();
  if (/missile|rocket|stinger|warhead|msl|pod_msl/.test(label)) return 'missile';
  if (/cape|cloak|mantle|wing|scarf|manto|mantm|_mant/.test(label)) return 'cape';
  if (/tail|(?:^|[^a-z])tl(?:[^a-z0-9]|$)/.test(label)) return 'tail';
  if (/pelvis|waist/.test(label)) return 'pelvis';
  if (/leg|thigh|shin|knee|calf|foot|feet|(?:^|[^a-z])lg(?:[^a-z0-9]|$)/.test(label)) return 'leg';
  if (/arm|hand|shld|shoulder|elbow|wrist|forearm|(?:^|[^a-z])ar(?:[^a-z0-9]|$)/.test(label)) return 'arm';
  if (/body|torso|chest|core|(?:^|[^a-z])bd(?:[^a-z0-9]|$)/.test(label)) return 'body';
  if (/head|helm|face|(?:^|[^a-z])hd(?:[^a-z0-9]|$)/.test(label)) return 'head';
  return null;
}

function isMeshUnderBuiltInWeapon(mesh: THREE.Mesh): boolean {
  let cur: THREE.Object3D | null = mesh;
  for (let depth = 0; cur && depth < 4; depth += 1) {
    if (isBuiltInWeaponNode(cur.name)) return true;
    cur = cur.parent;
  }
  return false;
}

function addChunkToList(
  list: TriChunk[],
  mesh: THREE.Mesh,
  toWrapper: THREE.Matrix4,
  tris: number[],
) {
  if (tris.length === 0) return;
  const existing = list.find((it) => it.mesh === mesh && it.toWrapper === toWrapper);
  if (existing) {
    existing.tris.push(...tris);
  } else {
    list.push({ mesh, toWrapper, tris: [...tris] });
  }
}

function splitClusterByYBands(
  mesh: THREE.Mesh,
  toWrapper: THREE.Matrix4,
  triIndices: number[],
  minY: number,
  spanY: number,
  highCut: number,
  midCut: number,
): {
  high: number[];
  highBox: THREE.Box3;
  mid: number[];
  midBox: THREE.Box3;
  low: number[];
  lowBox: THREE.Box3;
} {
  const high: number[] = [];
  const mid: number[] = [];
  const low: number[] = [];
  const highBox = new THREE.Box3();
  const midBox = new THREE.Box3();
  const lowBox = new THREE.Box3();
  const posAttr = mesh.geometry.getAttribute('position');
  const indexAttr = mesh.geometry.getIndex();
  if (!posAttr) return { high, highBox, mid, midBox, low, lowBox };

  const p0 = new THREE.Vector3();
  const p1 = new THREE.Vector3();
  const p2 = new THREE.Vector3();
  for (let i = 0; i < triIndices.length; i += 1) {
    const t = triIndices[i];
    const i0 = indexAttr ? indexAttr.getX(t * 3) : t * 3;
    const i1 = indexAttr ? indexAttr.getX(t * 3 + 1) : t * 3 + 1;
    const i2 = indexAttr ? indexAttr.getX(t * 3 + 2) : t * 3 + 2;
    p0.fromBufferAttribute(posAttr, i0).applyMatrix4(toWrapper);
    p1.fromBufferAttribute(posAttr, i1).applyMatrix4(toWrapper);
    p2.fromBufferAttribute(posAttr, i2).applyMatrix4(toWrapper);
    const cy = (p0.y + p1.y + p2.y) / 3;
    const yRatio = (cy - minY) / spanY;
    if (yRatio >= highCut) {
      high.push(t);
      highBox.expandByPoint(p0);
      highBox.expandByPoint(p1);
      highBox.expandByPoint(p2);
    } else if (yRatio >= midCut) {
      mid.push(t);
      midBox.expandByPoint(p0);
      midBox.expandByPoint(p1);
      midBox.expandByPoint(p2);
    } else {
      low.push(t);
      lowBox.expandByPoint(p0);
      lowBox.expandByPoint(p1);
      lowBox.expandByPoint(p2);
    }
  }
  return { high, highBox, mid, midBox, low, lowBox };
}

function splitGeometryClusters(
  geom: THREE.BufferGeometry,
  toWrapper: THREE.Matrix4,
): ExtractedCluster[] {
  const posAttr = geom.getAttribute('position');
  if (!posAttr) return [];
  const indexAttr = geom.getIndex();
  const triCount = indexAttr ? Math.floor(indexAttr.count / 3) : Math.floor(posAttr.count / 3);
  if (triCount <= 0) return [];

  const parent = new Int32Array(triCount);
  for (let i = 0; i < triCount; i++) parent[i] = i;
  const find = (x: number): number => {
    let r = x;
    while (parent[r] !== r) r = parent[r];
    let cur = x;
    while (cur !== r) {
      const nxt = parent[cur];
      parent[cur] = r;
      cur = nxt;
    }
    return r;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };

  const vertOwner = new Map<string, number>();
  const worldPositions: THREE.Vector3[] = new Array(triCount * 3);
  const v = new THREE.Vector3();

  for (let t = 0; t < triCount; t++) {
    for (let k = 0; k < 3; k++) {
      const idx = indexAttr ? indexAttr.getX(t * 3 + k) : t * 3 + k;
      v.fromBufferAttribute(posAttr, idx).applyMatrix4(toWrapper);
      worldPositions[t * 3 + k] = v.clone();
      const key = `${v.x.toFixed(3)},${v.y.toFixed(3)},${v.z.toFixed(3)}`;
      const existing = vertOwner.get(key);
      if (existing !== undefined) {
        union(t, existing);
      } else {
        vertOwner.set(key, t);
      }
    }
  }

  const map = new Map<number, ExtractedCluster>();
  for (let t = 0; t < triCount; t++) {
    const rootId = find(t);
    let cluster = map.get(rootId);
    if (!cluster) {
      cluster = { triIndices: [], center: new THREE.Vector3(), box: new THREE.Box3() };
      map.set(rootId, cluster);
    }
    cluster.triIndices.push(t);
    for (let k = 0; k < 3; k++) {
      cluster.box.expandByPoint(worldPositions[t * 3 + k]);
    }
  }

  const clusters = Array.from(map.values());
  clusters.forEach((c) => c.box.getCenter(c.center));
  return clusters;
}

function buildPivotSubMesh(
  sourceMesh: THREE.Mesh,
  toWrapper: THREE.Matrix4,
  triIndices: number[],
  pivotWorldPos: THREE.Vector3,
  geometriesOut: THREE.BufferGeometry[],
): THREE.Mesh | null {
  if (triIndices.length === 0) return null;
  const srcGeom = sourceMesh.geometry;
  const posAttr = srcGeom.getAttribute('position');
  const normAttr = srcGeom.getAttribute('normal');
  const uvAttr = srcGeom.getAttribute('uv');
  const indexAttr = srcGeom.getIndex();
  if (!posAttr) return null;

  const vertCount = triIndices.length * 3;
  const posArr = new Float32Array(vertCount * 3);
  const normArr = normAttr ? new Float32Array(vertCount * 3) : null;
  const uvArr = uvAttr ? new Float32Array(vertCount * 2) : null;

  const normalMat = new THREE.Matrix3().getNormalMatrix(toWrapper);
  const v = new THREE.Vector3();
  const n = new THREE.Vector3();

  let outIdx = 0;
  for (let i = 0; i < triIndices.length; i++) {
    const t = triIndices[i];
    for (let k = 0; k < 3; k++) {
      const srcIdx = indexAttr ? indexAttr.getX(t * 3 + k) : t * 3 + k;
      v.fromBufferAttribute(posAttr, srcIdx).applyMatrix4(toWrapper).sub(pivotWorldPos);
      posArr[outIdx * 3] = v.x;
      posArr[outIdx * 3 + 1] = v.y;
      posArr[outIdx * 3 + 2] = v.z;

      if (normAttr && normArr) {
        n.fromBufferAttribute(normAttr, srcIdx).applyMatrix3(normalMat).normalize();
        normArr[outIdx * 3] = n.x;
        normArr[outIdx * 3 + 1] = n.y;
        normArr[outIdx * 3 + 2] = n.z;
      }
      if (uvAttr && uvArr) {
        uvArr[outIdx * 2] = uvAttr.getX(srcIdx);
        uvArr[outIdx * 2 + 1] = uvAttr.getY(srcIdx);
      }
      outIdx++;
    }
  }

  const outGeom = new THREE.BufferGeometry();
  outGeom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  if (normArr) {
    outGeom.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));
  } else {
    outGeom.computeVertexNormals();
  }
  if (uvArr) {
    outGeom.setAttribute('uv', new THREE.BufferAttribute(uvArr, 2));
  }
  outGeom.computeBoundingBox();
  outGeom.computeBoundingSphere();
  geometriesOut.push(outGeom);

  const subMesh = new THREE.Mesh(outGeom, sourceMesh.material);
  subMesh.castShadow = true;
  subMesh.receiveShadow = true;
  return subMesh;
}

function tryArticulatePartRig(
  rawModel: THREE.Object3D,
  wrapper: THREE.Group,
  geometriesOut: THREE.BufferGeometry[],
  customCharacterId?: ModelStyle,
): {
  bones: CustomBoneSet;
  articulatedRoot: THREE.Group;
  handRCenterLocal?: THREE.Vector3;
  handLCenterLocal?: THREE.Vector3;
  backMissiles: THREE.Object3D[];
} | null {
  wrapper.updateMatrixWorld(true);
  const wrapperInv = wrapper.matrixWorld.clone().invert();

  const partMeshes: Partial<Record<PartTag, THREE.Mesh[]>> = {};
  const extraMeshes: THREE.Mesh[] = [];
  rawModel.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    if (isMeshUnderBuiltInWeapon(mesh)) return;
    const tag = getMeshPartTag(mesh);
    if (!tag) {
      extraMeshes.push(mesh);
      return;
    }
    if (!partMeshes[tag]) partMeshes[tag] = [];
    partMeshes[tag]!.push(mesh);
  });

  const matchedTags = Object.keys(partMeshes).length;
  if (matchedTags < 3 || !partMeshes.leg || !partMeshes.arm || !partMeshes.body) {
    return null;
  }

  const articulatedRoot = new THREE.Group();
  const wrapperOrigin = new THREE.Vector3(0, 0, 0);

  const getClusters = (tag: PartTag) => {
    const meshes = partMeshes[tag] ?? [];
    return meshes.map((m) => {
      const toWrapper = wrapperInv.clone().multiply(m.matrixWorld);
      return { mesh: m, toWrapper, clusters: splitGeometryClusters(m.geometry, toWrapper) };
    });
  };

  const wrapperBox = new THREE.Box3().setFromObject(wrapper);
  const wrapperHeight = Math.max(0.5, wrapperBox.max.y - wrapperBox.min.y);
  const hs = wrapperHeight / 2.08;

  const legItems = getClusters('leg');
  const armItems = getClusters('arm');
  const bodyItems = getClusters('body');
  const headItems = getClusters('head');
  const tailItems = getClusters('tail');
  const capeItems = getClusters('cape');
  const pelvisItems = getClusters('pelvis');
  const missileItems = getClusters('missile');

  // Split legs into pelvis, hipL/R, kneeL/R, footL/R
  const allLegBox = new THREE.Box3();
  legItems.forEach((item) => item.clusters.forEach((c) => allLegBox.union(c.box)));
  const legMinY = allLegBox.min.y;
  const legSpanY = Math.max(0.01, allLegBox.max.y - legMinY);

  const pelvisChunks: TriChunk[] = [];
  const createLegBucket = () => ({
    thigh: [] as TriChunk[],
    thighBox: new THREE.Box3(),
    shin: [] as TriChunk[],
    shinBox: new THREE.Box3(),
    foot: [] as TriChunk[],
    footBox: new THREE.Box3(),
  });
  const legBuckets = {
    L: createLegBucket(),
    R: createLegBucket(),
  };
  const pelvisBox = new THREE.Box3();

  pelvisItems.forEach((item) => {
    item.clusters.forEach((c) => {
      addChunkToList(pelvisChunks, item.mesh, item.toWrapper, c.triIndices);
      pelvisBox.union(c.box);
    });
  });

  legItems.forEach((item) => {
    item.clusters.forEach((c) => {
      const spansCenter =
        (c.box.min.x < -0.025 * hs && c.box.max.x > 0.025 * hs) ||
        Math.abs(c.center.x) < 0.05 * hs;
      if (spansCenter) {
        addChunkToList(pelvisChunks, item.mesh, item.toWrapper, c.triIndices);
        pelvisBox.union(c.box);
        return;
      }
      const side = c.center.x >= 0 ? 'L' : 'R';
      const bucket = legBuckets[side];
      const clusterHeight = c.box.max.y - c.box.min.y;
      if (clusterHeight > legSpanY * 0.68) {
        const sliced = splitClusterByYBands(
          item.mesh,
          item.toWrapper,
          c.triIndices,
          legMinY,
          legSpanY,
          0.46,
          0.15,
        );
        addChunkToList(bucket.thigh, item.mesh, item.toWrapper, sliced.high);
        if (!sliced.highBox.isEmpty()) bucket.thighBox.union(sliced.highBox);
        addChunkToList(bucket.shin, item.mesh, item.toWrapper, sliced.mid);
        if (!sliced.midBox.isEmpty()) bucket.shinBox.union(sliced.midBox);
        addChunkToList(bucket.foot, item.mesh, item.toWrapper, sliced.low);
        if (!sliced.lowBox.isEmpty()) bucket.footBox.union(sliced.lowBox);
        return;
      }
      const yRatio = (c.center.y - legMinY) / legSpanY;
      if (yRatio >= 0.46) {
        addChunkToList(bucket.thigh, item.mesh, item.toWrapper, c.triIndices);
        bucket.thighBox.union(c.box);
      } else if (yRatio >= 0.15) {
        addChunkToList(bucket.shin, item.mesh, item.toWrapper, c.triIndices);
        bucket.shinBox.union(c.box);
      } else {
        addChunkToList(bucket.foot, item.mesh, item.toWrapper, c.triIndices);
        bucket.footBox.union(c.box);
      }
    });
  });

  // Ensure each leg side always has a thigh root if any leg triangles exist on that side
  (['L', 'R'] as const).forEach((side) => {
    const b = legBuckets[side];
    if (b.thigh.length === 0 && b.shin.length > 0) {
      b.thigh = b.shin;
      b.thighBox = b.shinBox;
      b.shin = b.foot;
      b.shinBox = b.footBox;
      b.foot = [];
      b.footBox = new THREE.Box3();
    } else if (b.thigh.length === 0 && b.foot.length > 0) {
      b.thigh = b.foot;
      b.thighBox = b.footBox;
      b.foot = [];
      b.footBox = new THREE.Box3();
    }
  });

  // Pelvis meshes attached directly to articulatedRoot
  pelvisChunks.forEach((p) => {
    const m = buildPivotSubMesh(p.mesh, p.toWrapper, p.tris, wrapperOrigin, geometriesOut);
    if (m) articulatedRoot.add(m);
  });

  // Body (spineUpper) + detect any dorsal Hunter back-missiles
  const bodyBox = new THREE.Box3();
  let largestBodyTris = 0;
  bodyItems.forEach((item) =>
    item.clusters.forEach((c) => {
      bodyBox.union(c.box);
      if (c.triIndices.length > largestBodyTris) {
        largestBodyTris = c.triIndices.length;
      }
    }),
  );
  const bodyCenter = bodyBox.getCenter(new THREE.Vector3());
  const bodySpanY = Math.max(0.1, bodyBox.max.y - bodyBox.min.y);
  const waistY = !pelvisBox.isEmpty()
    ? Math.min(bodyBox.min.y + 0.14 * hs, pelvisBox.max.y - 0.05 * hs)
    : bodyBox.min.y + 0.08 * hs;
  const spinePivotWorld = new THREE.Vector3(0, waistY, (bodyBox.min.z + bodyBox.max.z) * 0.5);
  const spineGroup = new THREE.Group();
  spineGroup.position.copy(spinePivotWorld);
  articulatedRoot.add(spineGroup);

  const backMissiles: THREE.Object3D[] = [];
  const extractedBackClusters: Array<{
    mesh: THREE.Mesh;
    toWrapper: THREE.Matrix4;
    cluster: ExtractedCluster;
  }> = [];

  if (customCharacterId === 'hunter') {
    const candidates: Array<{
      itemIdx: number;
      clusterIdx: number;
      mesh: THREE.Mesh;
      toWrapper: THREE.Matrix4;
      cluster: ExtractedCluster;
    }> = [];
    const cSize = new THREE.Vector3();
    bodyItems.forEach((item, itemIdx) => {
      item.clusters.forEach((c, clusterIdx) => {
        c.box.getSize(cSize);
        if (
          c.triIndices.length >= 8 &&
          c.triIndices.length < largestBodyTris * 0.4 &&
          cSize.x < 0.36 * hs &&
          cSize.y < 0.48 * hs &&
          cSize.z < 0.44 * hs &&
          c.center.y > bodyBox.min.y + bodySpanY * 0.42 &&
          c.center.z < bodyCenter.z - 0.08 * hs &&
          c.box.max.z < bodyCenter.z + 0.04 * hs &&
          Math.abs(c.center.x) > 0.04 * hs
        ) {
          candidates.push({
            itemIdx,
            clusterIdx,
            mesh: item.mesh,
            toWrapper: item.toWrapper,
            cluster: c,
          });
        }
      });
    });

    if (candidates.length >= 2) {
      const skipSet = new Set<string>();
      candidates.forEach((cand) => {
        skipSet.add(`${cand.itemIdx}:${cand.clusterIdx}`);
        extractedBackClusters.push({
          mesh: cand.mesh,
          toWrapper: cand.toWrapper,
          cluster: cand.cluster,
        });
      });
      bodyItems.forEach((item, itemIdx) => {
        const torsoTris: number[] = [];
        item.clusters.forEach((c, clusterIdx) => {
          if (!skipSet.has(`${itemIdx}:${clusterIdx}`)) {
            torsoTris.push(...c.triIndices);
          }
        });
        const m = buildPivotSubMesh(item.mesh, item.toWrapper, torsoTris, spinePivotWorld, geometriesOut);
        if (m) spineGroup.add(m);
      });
    } else {
      bodyItems.forEach((item) => {
        const allTris = item.clusters.flatMap((c) => c.triIndices);
        const m = buildPivotSubMesh(item.mesh, item.toWrapper, allTris, spinePivotWorld, geometriesOut);
        if (m) spineGroup.add(m);
      });
    }
  } else {
    bodyItems.forEach((item) => {
      const allTris = item.clusters.flatMap((c) => c.triIndices);
      const m = buildPivotSubMesh(item.mesh, item.toWrapper, allTris, spinePivotWorld, geometriesOut);
      if (m) spineGroup.add(m);
    });
  }

  missileItems.forEach((item) => {
    item.clusters.forEach((c) => {
      extractedBackClusters.push({
        mesh: item.mesh,
        toWrapper: item.toWrapper,
        cluster: c,
      });
    });
  });

  extractedBackClusters.forEach(({ mesh, toWrapper, cluster }) => {
    const m = buildPivotSubMesh(mesh, toWrapper, cluster.triIndices, cluster.center, geometriesOut);
    if (m) {
      m.position.copy(cluster.center).sub(spinePivotWorld);
      spineGroup.add(m);
      backMissiles.push(m);
    }
  });

  // Head attached cleanly inside spineUpper's neck/collar socket (ignoring trailing rear helmet plumes/horns)
  let headGroup: THREE.Group | undefined;
  const headBox = new THREE.Box3();
  let headPivotWorld: THREE.Vector3 | undefined;
  if (headItems.length > 0) {
    headItems.forEach((item) => item.clusters.forEach((c) => headBox.union(c.box)));
    const headSize = new THREE.Vector3();
    const headCenter = new THREE.Vector3();
    headBox.getSize(headSize);
    headBox.getCenter(headCenter);

    const bodySize = bodyBox.getSize(new THREE.Vector3());
    const coreHeadBox = new THREE.Box3();
    if (!bodyBox.isEmpty()) {
      const hv = new THREE.Vector3();
      const xHalf = Math.max(0.18 * hs, bodySize.x * 0.52);
      const zMin = bodyBox.min.z - 0.06 * hs;
      const zMax = bodyBox.max.z + 0.14 * hs;
      headItems.forEach((item) => {
        const pos = item.mesh.geometry.getAttribute('position');
        if (!pos) return;
        for (let i = 0; i < pos.count; i += 1) {
          hv.fromBufferAttribute(pos, i).applyMatrix4(item.toWrapper);
          if (Math.abs(hv.x - bodyCenter.x) <= xHalf && hv.z >= zMin && hv.z <= zMax) {
            coreHeadBox.expandByPoint(hv);
          }
        }
      });
    }

    const refBox = !coreHeadBox.isEmpty() ? coreHeadBox : headBox;
    const refSize = refBox.getSize(new THREE.Vector3());
    const refCenter = refBox.getCenter(new THREE.Vector3());

    if (!bodyBox.isEmpty()) {
      const bodySpanZ = Math.max(0.1, bodySize.z);
      const rawNeckY = refBox.min.y + Math.min(0.08 * hs, refSize.y * 0.14);
      headPivotWorld = new THREE.Vector3(
        THREE.MathUtils.clamp(refCenter.x, bodyCenter.x - 0.03 * hs, bodyCenter.x + 0.03 * hs),
        THREE.MathUtils.clamp(
          rawNeckY,
          bodyBox.max.y - 0.16 * hs,
          Math.min(headBox.max.y - 0.12 * hs, bodyBox.max.y + 0.03 * hs),
        ),
        THREE.MathUtils.clamp(
          refCenter.z,
          bodyBox.min.z + bodySpanZ * 0.3,
          bodyBox.max.z - bodySpanZ * 0.3,
        ),
      );
    } else {
      headPivotWorld = new THREE.Vector3(
        headCenter.x,
        headBox.min.y + headSize.y * 0.14,
        headCenter.z,
      );
    }

    headGroup = new THREE.Group();
    headGroup.position.copy(headPivotWorld).sub(spinePivotWorld);
    spineGroup.add(headGroup);
    headItems.forEach((item) => {
      const allTris = item.clusters.flatMap((c) => c.triIndices);
      const m = buildPivotSubMesh(item.mesh, item.toWrapper, allTris, headPivotWorld!, geometriesOut);
      if (m) headGroup!.add(m);
    });
  }

  // Cape / Wing mantle attached to spineUpper shoulders
  const capeLGroups: THREE.Group[] = [];
  const capeRGroups: THREE.Group[] = [];
  if (capeItems.length > 0) {
    const allCapeClusters: Array<{ mesh: THREE.Mesh; toWrapper: THREE.Matrix4; cluster: ExtractedCluster }> = [];
    capeItems.forEach((item) => {
      item.clusters.forEach((c) => allCapeClusters.push({ mesh: item.mesh, toWrapper: item.toWrapper, cluster: c }));
    });
    const leftSeparated = allCapeClusters.filter(
      (it) => it.cluster.center.x > 0.08 * hs && it.cluster.box.min.x > -0.02 * hs,
    );
    const rightSeparated = allCapeClusters.filter(
      (it) => it.cluster.center.x < -0.08 * hs && it.cluster.box.max.x < 0.02 * hs,
    );
    const hasSplitWings = leftSeparated.length > 0 && rightSeparated.length > 0 && leftSeparated.length + rightSeparated.length === allCapeClusters.length;

    const buildCapePivot = (subset: typeof allCapeClusters): THREE.Group | null => {
      if (subset.length === 0) return null;
      const capeBox = new THREE.Box3();
      subset.forEach((it) => capeBox.union(it.cluster.box));
      const capeSize = new THREE.Vector3();
      const capeCenter = new THREE.Vector3();
      capeBox.getSize(capeSize);
      capeBox.getCenter(capeCenter);
      const capePivotWorld = new THREE.Vector3(
        capeCenter.x,
        capeBox.max.y - capeSize.y * 0.08,
        capeBox.max.z - Math.min(0.08 * hs, capeSize.z * 0.12),
      );
      const grp = new THREE.Group();
      grp.position.copy(capePivotWorld).sub(spinePivotWorld);
      spineGroup.add(grp);
      subset.forEach((it) => {
        const m = buildPivotSubMesh(it.mesh, it.toWrapper, it.cluster.triIndices, capePivotWorld, geometriesOut);
        if (m) grp.add(m);
      });
      return grp;
    };

    if (hasSplitWings) {
      const lGrp = buildCapePivot(leftSeparated);
      const rGrp = buildCapePivot(rightSeparated);
      if (lGrp) capeLGroups.push(lGrp);
      if (rGrp) capeRGroups.push(rGrp);
    } else {
      const singleCapeGrp = buildCapePivot(allCapeClusters);
      if (singleCapeGrp) capeLGroups.push(singleCapeGrp);
    }
  }

  // Tail attached to articulatedRoot
  let tailGroup: THREE.Group | undefined;
  if (tailItems.length > 0) {
    const tailBox = new THREE.Box3();
    tailItems.forEach((item) => item.clusters.forEach((c) => tailBox.union(c.box)));
    const tailSize = new THREE.Vector3();
    tailBox.getSize(tailSize);
    const tailPivotWorld = new THREE.Vector3(
      0,
      tailBox.max.y - tailSize.y * 0.08,
      tailBox.max.z - Math.min(0.12 * hs, tailSize.z * 0.08),
    );
    tailGroup = new THREE.Group();
    tailGroup.position.copy(tailPivotWorld);
    articulatedRoot.add(tailGroup);
    tailItems.forEach((item) => {
      const allTris = item.clusters.flatMap((c) => c.triIndices);
      const m = buildPivotSubMesh(item.mesh, item.toWrapper, allTris, tailPivotWorld, geometriesOut);
      if (m) tailGroup!.add(m);
    });
  }

  // Split arms into shoulder -> elbow -> hand on each side
  const allArmBox = new THREE.Box3();
  armItems.forEach((item) => item.clusters.forEach((c) => allArmBox.union(c.box)));
  const armMinY = allArmBox.min.y;
  const armSpanY = Math.max(0.01, allArmBox.max.y - armMinY);

  const createArmBucket = () => ({
    upper: [] as TriChunk[],
    upperBox: new THREE.Box3(),
    fore: [] as TriChunk[],
    foreBox: new THREE.Box3(),
    hand: [] as TriChunk[],
    handBox: new THREE.Box3(),
  });
  const armBuckets = {
    L: createArmBucket(),
    R: createArmBucket(),
  };

  armItems.forEach((item) => {
    item.clusters.forEach((c) => {
      const side = c.center.x >= 0 ? 'L' : 'R';
      const bucket = armBuckets[side];
      const clusterHeight = c.box.max.y - c.box.min.y;
      if (clusterHeight > armSpanY * 0.68) {
        const sliced = splitClusterByYBands(
          item.mesh,
          item.toWrapper,
          c.triIndices,
          armMinY,
          armSpanY,
          0.62,
          0.21,
        );
        addChunkToList(bucket.upper, item.mesh, item.toWrapper, sliced.high);
        if (!sliced.highBox.isEmpty()) bucket.upperBox.union(sliced.highBox);
        addChunkToList(bucket.fore, item.mesh, item.toWrapper, sliced.mid);
        if (!sliced.midBox.isEmpty()) bucket.foreBox.union(sliced.midBox);
        addChunkToList(bucket.hand, item.mesh, item.toWrapper, sliced.low);
        if (!sliced.lowBox.isEmpty()) bucket.handBox.union(sliced.lowBox);
        return;
      }
      const yRatio = (c.center.y - armMinY) / armSpanY;
      if (yRatio >= 0.62) {
        addChunkToList(bucket.upper, item.mesh, item.toWrapper, c.triIndices);
        bucket.upperBox.union(c.box);
      } else if (yRatio >= 0.21) {
        addChunkToList(bucket.fore, item.mesh, item.toWrapper, c.triIndices);
        bucket.foreBox.union(c.box);
      } else {
        addChunkToList(bucket.hand, item.mesh, item.toWrapper, c.triIndices);
        bucket.handBox.union(c.box);
      }
    });
  });

  // Ensure each arm side always has an upper shoulder root if any arm triangles exist on that side
  (['L', 'R'] as const).forEach((side) => {
    const b = armBuckets[side];
    if (b.upper.length === 0 && b.fore.length > 0) {
      b.upper = b.fore;
      b.upperBox = b.foreBox;
      b.fore = b.hand;
      b.foreBox = b.handBox;
      b.hand = [];
      b.handBox = new THREE.Box3();
    } else if (b.upper.length === 0 && b.hand.length > 0) {
      b.upper = b.hand;
      b.upperBox = b.handBox;
      b.hand = [];
      b.handBox = new THREE.Box3();
    }
  });

  const appendChunksToGroup = (
    chunks: TriChunk[],
    pivotWorld: THREE.Vector3,
    targetGroup: THREE.Group,
  ) => {
    chunks.forEach((ch) => {
      const m = buildPivotSubMesh(ch.mesh, ch.toWrapper, ch.tris, pivotWorld, geometriesOut);
      if (m) targetGroup.add(m);
    });
  };

  const buildArmChain = (side: 'L' | 'R') => {
    const b = armBuckets[side];
    if (b.upper.length === 0) return {};
    const uCenter = b.upperBox.getCenter(new THREE.Vector3());
    const uSize = b.upperBox.getSize(new THREE.Vector3());
    const isHeavyArm = customCharacterId === 'emperor';
    const xInset = isHeavyArm ? 0.2 : 0.28;
    const yDrop = isHeavyArm ? 0.34 : 0.16;
    const shoulderZ =
      isHeavyArm && !bodyBox.isEmpty()
        ? THREE.MathUtils.clamp(
            uCenter.z,
            bodyBox.min.z + Math.max(0.08 * hs, (bodyBox.max.z - bodyBox.min.z) * 0.28),
            bodyBox.max.z - Math.max(0.08 * hs, (bodyBox.max.z - bodyBox.min.z) * 0.28),
          )
        : uCenter.z;
    const shoulderWorld = new THREE.Vector3(
      side === 'L' ? b.upperBox.min.x + uSize.x * xInset : b.upperBox.max.x - uSize.x * xInset,
      b.upperBox.max.y - uSize.y * yDrop,
      shoulderZ,
    );
    const shoulderGrp = new THREE.Group();
    shoulderGrp.position.copy(shoulderWorld).sub(spinePivotWorld);
    spineGroup.add(shoulderGrp);
    appendChunksToGroup(b.upper, shoulderWorld, shoulderGrp);

    let elbowGrp: THREE.Group | undefined;
    let handGrp: THREE.Group | undefined;
    let handCenterLocal: THREE.Vector3 | undefined;
    if (b.fore.length > 0) {
      const fCenter = b.foreBox.getCenter(new THREE.Vector3());
      const elbowWorld = new THREE.Vector3(fCenter.x, b.foreBox.max.y - 0.04 * hs, fCenter.z);
      elbowGrp = new THREE.Group();
      elbowGrp.position.copy(elbowWorld).sub(shoulderWorld);
      shoulderGrp.add(elbowGrp);
      appendChunksToGroup(b.fore, elbowWorld, elbowGrp);

      if (b.hand.length > 0) {
        const hCenter = b.handBox.getCenter(new THREE.Vector3());
        const hSize = b.handBox.getSize(new THREE.Vector3());
        const handWorld = new THREE.Vector3(hCenter.x, b.handBox.max.y - 0.03 * hs, hCenter.z);
        handGrp = new THREE.Group();
        handGrp.position.copy(handWorld).sub(elbowWorld);
        elbowGrp.add(handGrp);
        appendChunksToGroup(b.hand, handWorld, handGrp);
        if (customCharacterId === 'emperor') {
          const fistAnchor = new THREE.Vector3(
            hCenter.x,
            b.handBox.min.y + hSize.y * 0.34,
            b.handBox.min.z + hSize.z * 0.58,
          );
          handCenterLocal = fistAnchor.sub(handWorld);
        } else {
          handCenterLocal = hCenter.clone().sub(handWorld);
        }
      }
    }
    return { shoulderGrp, elbowGrp, handGrp, handCenterLocal };
  };

  const buildLegChain = (side: 'L' | 'R') => {
    const b = legBuckets[side];
    if (b.thigh.length === 0) return {};
    const tCenter = b.thighBox.getCenter(new THREE.Vector3());
    const hipWorld = new THREE.Vector3(tCenter.x, b.thighBox.max.y - 0.05 * hs, tCenter.z);
    const hipGrp = new THREE.Group();
    hipGrp.position.copy(hipWorld);
    articulatedRoot.add(hipGrp);
    appendChunksToGroup(b.thigh, hipWorld, hipGrp);

    let kneeGrp: THREE.Group | undefined;
    let footGrp: THREE.Group | undefined;
    if (b.shin.length > 0) {
      const sCenter = b.shinBox.getCenter(new THREE.Vector3());
      const kneeWorld = new THREE.Vector3(sCenter.x, b.shinBox.max.y - 0.05 * hs, sCenter.z);
      kneeGrp = new THREE.Group();
      kneeGrp.position.copy(kneeWorld).sub(hipWorld);
      hipGrp.add(kneeGrp);
      appendChunksToGroup(b.shin, kneeWorld, kneeGrp);

      if (b.foot.length > 0) {
        const fCenter = b.footBox.getCenter(new THREE.Vector3());
        const footWorld = new THREE.Vector3(fCenter.x, b.footBox.max.y - 0.04 * hs, fCenter.z * 0.45);
        footGrp = new THREE.Group();
        footGrp.position.copy(footWorld).sub(kneeWorld);
        kneeGrp.add(footGrp);
        appendChunksToGroup(b.foot, footWorld, footGrp);
      }
    }
    return { hipGrp, kneeGrp, footGrp };
  };

  const armL = buildArmChain('L');
  const armR = buildArmChain('R');
  const legL = buildLegChain('L');
  const legR = buildLegChain('R');

  // Attach any extra decorative non-weapon meshes (crests, collars, belts) cleanly to the closest parent pivot
  extraMeshes.forEach((m) => {
    const toWrapper = wrapperInv.clone().multiply(m.matrixWorld);
    const clusters = splitGeometryClusters(m.geometry, toWrapper);
    if (clusters.length === 0) return;
    const allTris = clusters.flatMap((c) => c.triIndices);
    const exBox = new THREE.Box3();
    clusters.forEach((c) => exBox.union(c.box));
    const exCenter = exBox.getCenter(new THREE.Vector3());

    let parentGrp: THREE.Group = articulatedRoot;
    let parentPivotWorld = wrapperOrigin;
    const headAttachY = headPivotWorld
      ? Math.max(headPivotWorld.y - 0.02 * hs, bodyBox.max.y - 0.06 * hs)
      : headBox.min.y - 0.04 * hs;
    if (
      headGroup &&
      !headBox.isEmpty() &&
      exCenter.y >= headAttachY &&
      Math.abs(exCenter.x - bodyCenter.x) <= Math.max(0.28 * hs, (bodyBox.max.x - bodyBox.min.x) * 0.6)
    ) {
      parentGrp = headGroup;
      parentPivotWorld = new THREE.Vector3().addVectors(spinePivotWorld, headGroup.position);
    } else if (exCenter.y >= spinePivotWorld.y) {
      parentGrp = spineGroup;
      parentPivotWorld = spinePivotWorld;
    }
    const sub = buildPivotSubMesh(m, toWrapper, allTris, parentPivotWorld, geometriesOut);
    if (sub) {
      parentGrp.add(sub);
      m.visible = false;
    }
  });

  // Hide original rigid meshes now that articulated sub-meshes replace them
  Object.values(partMeshes).forEach((arr) => {
    arr?.forEach((m) => {
      m.visible = false;
    });
  });

  articulatedRoot.updateMatrixWorld(true);
  const makeIdentityBound = (node: THREE.Object3D): BoundBone => ({
    node,
    restLocalQuat: new THREE.Quaternion(),
    parentInCharQuat: new THREE.Quaternion(),
  });

  const bones: CustomBoneSet = {
    spineUpper: makeIdentityBound(spineGroup),
    head: headGroup ? makeIdentityBound(headGroup) : undefined,
    tail: tailGroup ? makeIdentityBound(tailGroup) : undefined,
    shoulderL: armL.shoulderGrp ? makeIdentityBound(armL.shoulderGrp) : undefined,
    elbowL: armL.elbowGrp ? makeIdentityBound(armL.elbowGrp) : undefined,
    handL: armL.handGrp ? makeIdentityBound(armL.handGrp) : undefined,
    shoulderR: armR.shoulderGrp ? makeIdentityBound(armR.shoulderGrp) : undefined,
    elbowR: armR.elbowGrp ? makeIdentityBound(armR.elbowGrp) : undefined,
    handR: armR.handGrp ? makeIdentityBound(armR.handGrp) : undefined,
    hipL: legL.hipGrp ? makeIdentityBound(legL.hipGrp) : undefined,
    kneeL: legL.kneeGrp ? makeIdentityBound(legL.kneeGrp) : undefined,
    footL: legL.footGrp ? makeIdentityBound(legL.footGrp) : undefined,
    hipR: legR.hipGrp ? makeIdentityBound(legR.hipGrp) : undefined,
    kneeR: legR.kneeGrp ? makeIdentityBound(legR.kneeGrp) : undefined,
    footR: legR.footGrp ? makeIdentityBound(legR.footGrp) : undefined,
    capeL: capeLGroups.map((g) => makeIdentityBound(g)),
    capeR: capeRGroups.map((g) => makeIdentityBound(g)),
  };

  return {
    bones,
    articulatedRoot,
    handRCenterLocal: armR.handCenterLocal,
    handLCenterLocal: armL.handCenterLocal,
    backMissiles,
  };
}

const NEUTRAL_PLACEHOLDER_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGM4ffUhAATwAoLSC4yKAAAAAElFTkSuQmCC';

function deduplicateColladaNodeIds(xmlText: string): string {
  if (typeof DOMParser === 'undefined' || typeof XMLSerializer === 'undefined') {
    return xmlText;
  }
  const doc = new DOMParser().parseFromString(xmlText, 'application/xml');
  if (doc.getElementsByTagName('parsererror').length > 0) {
    return xmlText;
  }
  const nodes = doc.getElementsByTagName('node');
  const seen = new Set<string>();
  let changed = false;
  let dupIdx = 1;
  for (let i = 0; i < nodes.length; i += 1) {
    const el = nodes[i];
    const id = el.getAttribute('id');
    if (!id) continue;
    if (seen.has(id)) {
      let nextId = `${id}__dup_${dupIdx++}`;
      while (seen.has(nextId)) {
        nextId = `${id}__dup_${dupIdx++}`;
      }
      el.setAttribute('id', nextId);
      seen.add(nextId);
      changed = true;
    } else {
      seen.add(id);
    }
  }
  return changed ? new XMLSerializer().serializeToString(doc) : xmlText;
}

function buildBlobResourceManager(textures?: Record<string, ArrayBuffer>): {
  manager: THREE.LoadingManager;
  waitPending: (timeoutMs?: number) => Promise<void>;
  disposeResources: () => void;
} {
  const manager = new THREE.LoadingManager();
  const blobUrls: string[] = [];
  const urlByBaseName = new Map<string, string>();
  let activeCount = 0;
  let loadedTriggered = false;

  if (textures) {
    for (const [rawName, buf] of Object.entries(textures)) {
      const base = (rawName.split(/[/\\]/).pop() || rawName).toLowerCase();
      const mime = base.endsWith('.png')
        ? 'image/png'
        : base.endsWith('.jpg') || base.endsWith('.jpeg')
          ? 'image/jpeg'
          : base.endsWith('.webp')
            ? 'image/webp'
            : 'application/octet-stream';
      const blobUrl = URL.createObjectURL(new Blob([buf], { type: mime }));
      blobUrls.push(blobUrl);
      urlByBaseName.set(base, blobUrl);
      urlByBaseName.set(base.replace(/\s+/g, '_'), blobUrl);
    }
  }

  let onLoadResolve: (() => void) | null = null;
  manager.onStart = () => {
    activeCount += 1;
  };
  manager.onLoad = () => {
    loadedTriggered = true;
    activeCount = 0;
    onLoadResolve?.();
    onLoadResolve = null;
  };
  manager.onError = () => {
    activeCount = Math.max(0, activeCount - 1);
  };

  manager.setURLModifier((url) => {
    if (
      !url ||
      url.startsWith('blob:') ||
      url.startsWith('data:') ||
      url.startsWith('http://') ||
      url.startsWith('https://')
    ) {
      return url;
    }
    const clean = url.split('#')[0].split('?')[0];
    let decoded = clean;
    try {
      decoded = decodeURIComponent(clean);
    } catch {
      // Keep raw if malformed URI encoding
    }
    const base = (decoded.split(/[/\\]/).pop() || decoded).toLowerCase();
    const mapped =
      urlByBaseName.get(base) ?? urlByBaseName.get(base.replace(/\s+/g, '_'));
    if (mapped) {
      return mapped;
    }
    if (/\.(png|jpe?g|webp|bmp|gif)$/i.test(base)) {
      return NEUTRAL_PLACEHOLDER_PNG;
    }
    return url;
  });

  const waitPending = (timeoutMs = 700) => {
    if (activeCount === 0 || loadedTriggered) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => {
      const timer = window.setTimeout(() => {
        onLoadResolve = null;
        resolve();
      }, timeoutMs);
      onLoadResolve = () => {
        window.clearTimeout(timer);
        resolve();
      };
    });
  };

  const disposeResources = () => {
    blobUrls.forEach((u) => URL.revokeObjectURL(u));
    blobUrls.length = 0;
    urlByBaseName.clear();
  };

  return { manager, waitPending, disposeResources };
}

async function parseModelObject(
  asset: Pick<CustomModelAsset, 'ext' | 'buffer' | 'textures'>,
): Promise<{
  object: THREE.Object3D;
  animations: THREE.AnimationClip[];
  disposeResources: () => void;
}> {
  const { ext, buffer, textures } = asset;
  // Clone buffer slice so parsers that transfer/detach don't mutate cached buffer
  const copy = buffer.slice(0);
  const { manager, waitPending, disposeResources } = buildBlobResourceManager(textures);

  if (ext === '.glb' || ext === '.gltf') {
    const loader = new GLTFLoader(manager);
    return new Promise((resolve, reject) => {
      loader.parse(
        copy,
        '',
        async (gltf) => {
          await waitPending();
          resolve({
            object: gltf.scene,
            animations: gltf.animations ?? [],
            disposeResources,
          });
        },
        (err) => {
          disposeResources();
          reject(err);
        },
      );
    });
  }

  if (ext === '.dae') {
    const rawText = new TextDecoder().decode(copy);
    const text = deduplicateColladaNodeIds(rawText);
    const loader = new ColladaLoader(manager);
    const collada = loader.parse(text, '');
    if (!collada || !collada.scene) {
      disposeResources();
      return Promise.reject(new Error('Не удалось разобрать 3D-сцену из файла .dae'));
    }
    await waitPending();
    return {
      object: collada.scene,
      animations: collada.scene.animations ?? [],
      disposeResources,
    };
  }

  if (ext === '.fbx') {
    const loader = new FBXLoader(manager);
    const obj = loader.parse(copy, '');
    await waitPending();
    return {
      object: obj,
      animations:
        (obj as THREE.Object3D & { animations?: THREE.AnimationClip[] })
          .animations ?? [],
      disposeResources,
    };
  }

  if (ext === '.obj') {
    const text = new TextDecoder().decode(copy);
    const loader = new OBJLoader(manager);
    if (textures) {
      const mtlEntry = Object.entries(textures).find(([name]) =>
        name.toLowerCase().endsWith('.mtl'),
      );
      if (mtlEntry) {
        const mtlText = new TextDecoder().decode(mtlEntry[1]);
        const mtlLoader = new MTLLoader(manager);
        const materials = mtlLoader.parse(mtlText, '');
        materials.preload();
        loader.setMaterials(materials);
      }
    }
    const obj = loader.parse(text);
    await waitPending();
    return { object: obj, animations: [], disposeResources };
  }

  if (ext === '.stl') {
    disposeResources();
    const loader = new STLLoader();
    const geometry = loader.parse(copy);
    geometry.computeVertexNormals();
    const mat = new THREE.MeshPhysicalMaterial({
      color: '#2563EB',
      metalness: 0.55,
      roughness: 0.25,
      clearcoat: 0.8,
    });
    const mesh = new THREE.Mesh(geometry, mat);
    const grp = new THREE.Group();
    grp.add(mesh);
    return { object: grp, animations: [], disposeResources: () => {} };
  }

  disposeResources();
  return Promise.reject(new Error(`Неподдерживаемый формат: ${ext}`));
}

const ACHILLES_FALLBACK_CROWN_HEIGHT = 1.76;
let cachedAchillesCrown: { key: string; height: number } | null = null;
let pendingAchillesCrown: { key: string; promise: Promise<number> } | null = null;

function achillesCacheKey(asset: CustomModelAsset): string {
  return `${asset.filename}:${asset.size}:${asset.yawOffsetDeg}:${asset.playAnimation ? 1 : 0}`;
}

function collectModelBodyVerticesAndBox(
  rawModel: THREE.Object3D,
  excludeWeapons: boolean,
  ysOut?: number[],
): THREE.Box3 {
  const out = new THREE.Box3();
  const v = new THREE.Vector3();
  rawModel.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    if (excludeWeapons && isMeshUnderBuiltInWeapon(mesh)) return;
    const posAttr = mesh.geometry.getAttribute('position');
    if (!posAttr) return;
    const skinned = mesh as THREE.SkinnedMesh;
    if (skinned.isSkinnedMesh) {
      skinned.skeleton?.update();
    }
    const step = posAttr.count > 25000 ? Math.ceil(posAttr.count / 25000) : 1;
    for (let i = 0; i < posAttr.count; i += step) {
      if (skinned.isSkinnedMesh) {
        skinned.getVertexPosition(i, v);
      } else {
        v.fromBufferAttribute(posAttr, i);
      }
      v.applyMatrix4(mesh.matrixWorld);
      out.expandByPoint(v);
      if (ysOut) ysOut.push(v.y);
    }
  });
  return out;
}

function estimateAchillesCrownHeightFromPosed(
  rawModel: THREE.Object3D,
  wrapper: THREE.Group,
  bones: CustomBoneSet,
): number {
  wrapper.updateMatrixWorld(true);
  const ys: number[] = [];
  let box = collectModelBodyVerticesAndBox(rawModel, true, ys);
  if (box.isEmpty() || ys.length === 0) {
    ys.length = 0;
    box = collectModelBodyVerticesAndBox(rawModel, false, ys);
  }
  if (box.isEmpty() || ys.length === 0) {
    return ACHILLES_FALLBACK_CROWN_HEIGHT;
  }

  const minY = box.min.y;
  const totalH = Math.max(0.1, box.max.y - minY);
  ys.sort((a, b) => a - b);
  const p92H = ys[Math.floor((ys.length - 1) * 0.92)] - minY;
  const candidates: number[] = [p92H * 1.03];

  const tmpPos = new THREE.Vector3();
  if (bones.head) {
    bones.head.node.getWorldPosition(tmpPos);
    const headH = tmpPos.y - minY;
    if (headH > totalH * 0.45 && headH < totalH * 0.95) {
      candidates.push(headH * 1.22);
    }
  }
  if (bones.shoulderR) {
    bones.shoulderR.node.getWorldPosition(tmpPos);
    const shH = tmpPos.y - minY;
    if (shH > totalH * 0.4 && shH < totalH * 0.9) {
      candidates.push(shH * 1.36);
    }
  }

  const avg = candidates.reduce((acc, val) => acc + val, 0) / candidates.length;
  return THREE.MathUtils.clamp(Math.min(totalH, avg), 1.54, 1.82);
}

async function getAchillesReferenceCrownHeight(): Promise<number> {
  const achAsset = getCachedCustomModel('achilles');
  if (!achAsset) {
    return ACHILLES_FALLBACK_CROWN_HEIGHT;
  }
  const key = achillesCacheKey(achAsset);
  if (cachedAchillesCrown && cachedAchillesCrown.key === key) {
    return cachedAchillesCrown.height;
  }
  if (pendingAchillesCrown && pendingAchillesCrown.key === key) {
    return pendingAchillesCrown.promise;
  }

  const promise = (async () => {
    let disposeResources: (() => void) | null = null;
    try {
      const parsed = await parseModelObject({
        ext: achAsset.ext,
        buffer: achAsset.buffer,
        textures: {},
      });
      disposeResources = parsed.disposeResources;
      const raw = parsed.object;

      const wrapper = new THREE.Group();
      const yawGroup = new THREE.Group();
      yawGroup.add(raw);
      wrapper.add(yawGroup);
      yawGroup.rotation.y = THREE.MathUtils.degToRad(achAsset.yawOffsetDeg || 0);
      wrapper.updateMatrixWorld(true);

      let box = collectModelBodyVerticesAndBox(raw, true);
      if (box.isEmpty()) box = collectModelBodyVerticesAndBox(raw, false);
      if (box.isEmpty()) box = new THREE.Box3().setFromObject(wrapper);
      const size = box.getSize(new THREE.Vector3());
      const isUpright = size.y >= Math.max(size.x, size.z) * 0.45;
      const refHeight = isUpright
        ? Math.max(size.y, 0.001)
        : Math.max(size.y, size.x * 0.75, size.z * 0.75, 0.001);
      const scale = 2.08 / refHeight;
      yawGroup.scale.multiplyScalar(scale);
      wrapper.updateMatrixWorld(true);

      let scaledBox = collectModelBodyVerticesAndBox(raw, true);
      if (scaledBox.isEmpty()) scaledBox = collectModelBodyVerticesAndBox(raw, false);
      if (scaledBox.isEmpty()) scaledBox = new THREE.Box3().setFromObject(wrapper);
      const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
      yawGroup.position.x -= scaledCenter.x;
      yawGroup.position.z -= scaledCenter.z;
      yawGroup.position.y -= scaledBox.min.y;
      wrapper.updateMatrixWorld(true);

      const bones = bindSkeletonBones(raw, wrapper);
      if (achAsset.playAnimation && parsed.animations.length > 0) {
        const mixer = new THREE.AnimationMixer(raw);
        const clip = parsed.animations[0];
        if (clip) {
          mixer.clipAction(clip).play();
          mixer.update(0.15);
        }
      } else if (bones.shoulderR || bones.spineLower) {
        const ready = buildCustomLancePose({ mode: 'ready', elapsed: 0 });
        if (bones.spineLower) {
          rotateBoneInCharSpace(bones.spineLower, ready.torsoPitch * 0.45, ready.torsoYaw * 0.55, ready.torsoRoll * 0.45);
          rotateBoneInCharSpace(bones.spineUpper, ready.torsoPitch * 0.55, ready.torsoYaw * 0.45, ready.torsoRoll * 0.55);
        } else {
          rotateBoneInCharSpace(bones.spineUpper, ready.torsoPitch, ready.torsoYaw, ready.torsoRoll);
        }
        rotateBoneInCharSpace(bones.head, ready.headPitch, ready.headYaw, 0);
        rotateBoneInCharSpace(bones.hipL, ready.lHipX, 0, ready.lHipZ);
        rotateBoneInCharSpace(bones.kneeL, ready.lKneeX, 0, 0);
        rotateBoneInCharSpace(bones.hipR, ready.rHipX, 0, ready.rHipZ);
        rotateBoneInCharSpace(bones.kneeR, ready.rKneeX, 0, 0);
      }

      const crown = estimateAchillesCrownHeightFromPosed(raw, wrapper, bones);
      raw.traverse((ch) => {
        const m = ch as THREE.Mesh;
        if (m.isMesh && m.geometry) m.geometry.dispose();
      });
      cachedAchillesCrown = { key, height: crown };
      return crown;
    } catch {
      return ACHILLES_FALLBACK_CROWN_HEIGHT;
    } finally {
      disposeResources?.();
      if (pendingAchillesCrown?.key === key) {
        pendingAchillesCrown = null;
      }
    }
  })();

  pendingAchillesCrown = { key, promise };
  return promise;
}

export async function buildCustomLbxRig(
  asset: CustomModelAsset,
  loadout: LbxLoadout,
): Promise<CustomLbxRig> {
  const { object: rawModel, animations, disposeResources } =
    await parseModelObject(asset);
  const weapon = getWeaponById(loadout.weaponId);

  const root = new THREE.Group();
  const bodyGroup = new THREE.Group();
  root.add(bodyGroup);

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];
  const textures: THREE.Texture[] = [];
  const hitFlashMaterials: THREE.MeshStandardMaterial[] = [];
  let extraDispose: (() => void) | null = null;

  // Prepare materials, shadows, and collect built-in weapon nodes
  let hasBuiltInWeapon = false;
  const weaponNodes: THREE.Object3D[] = [];

  rawModel.traverse((child) => {
    if (isBuiltInWeaponNode(child.name)) {
      hasBuiltInWeapon = true;
      weaponNodes.push(child);
    }

    const mesh = child as THREE.Mesh;
    if (mesh.isMesh) {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      if (mesh.geometry) geometries.push(mesh.geometry);

      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((m) => {
        if (!m) return;
        materials.push(m);
        const isOutlineMat = /^line(_?m)?$/i.test(m.name || '');
        if ((m as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
          const std = m as THREE.MeshStandardMaterial;
          std.envMapIntensity = Math.max(std.envMapIntensity ?? 1, 0.85);
          if (!isOutlineMat) hitFlashMaterials.push(std);
        } else if (
          !isOutlineMat &&
          ((m as THREE.MeshPhongMaterial).isMeshPhongMaterial ||
            (m as THREE.MeshLambertMaterial).isMeshLambertMaterial)
        ) {
          hitFlashMaterials.push(m as unknown as THREE.MeshStandardMaterial);
        }
      });
    }
  });

  const customCharacterId: ModelStyle = asset.characterId ?? 'achilles';
  const weaponNodeSet = new Set<THREE.Object3D>(weaponNodes);
  const isWeaponMesh = (obj: THREE.Object3D): boolean => {
    let cur: THREE.Object3D | null = obj;
    while (cur && cur !== rawModel) {
      if (weaponNodeSet.has(cur) || isBuiltInWeaponNode(cur.name)) return true;
      cur = cur.parent;
    }
    return false;
  };

  const _tmpVert = new THREE.Vector3();
  const computeBodyBox = (includeWeapons: boolean): THREE.Box3 => {
    const out = new THREE.Box3();
    rawModel.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      if (!includeWeapons && isWeaponMesh(mesh)) return;
      const posAttr = mesh.geometry.getAttribute('position');
      if (!posAttr) return;
      const skinned = mesh as THREE.SkinnedMesh;
      if (skinned.isSkinnedMesh) {
        skinned.skeleton?.update();
      }
      const step = posAttr.count > 25000 ? Math.ceil(posAttr.count / 25000) : 1;
      for (let i = 0; i < posAttr.count; i += step) {
        if (skinned.isSkinnedMesh) {
          skinned.getVertexPosition(i, _tmpVert);
        } else {
          _tmpVert.fromBufferAttribute(posAttr, i);
        }
        _tmpVert.applyMatrix4(mesh.matrixWorld);
        out.expandByPoint(_tmpVert);
      }
    });
    return out;
  };

  // Determine target height anchored to Achilles' visual crown height so Hunter & Kunoichi match Achilles
  let targetHeight = 2.08;
  if (customCharacterId !== 'achilles') {
    const achCrown = await getAchillesReferenceCrownHeight();
    if (customCharacterId === 'hunter') {
      targetHeight = Math.max(1.62, achCrown * 1.03);
    } else if (customCharacterId === 'kunoichi') {
      targetHeight = Math.max(1.66, achCrown * 1.06);
    } else if (customCharacterId === 'emperor') {
      targetHeight = Math.max(1.74, achCrown * 1.1);
    } else {
      targetHeight = Math.max(1.72, achCrown * 1.08);
    }
  }
  const heightScale = targetHeight / 2.08;

  // Ground contact shadow
  const shadowTex = createContactShadowTexture();
  textures.push(shadowTex);
  const shadowGeo = new THREE.PlaneGeometry(1.9, 1.9);
  const shadowMat = new THREE.MeshBasicMaterial({
    map: shadowTex,
    transparent: true,
    depthWrite: false,
    opacity: 0.65,
  });
  geometries.push(shadowGeo);
  materials.push(shadowMat);
  const shadowDisc = new THREE.Mesh(shadowGeo, shadowMat);
  shadowDisc.rotation.x = -Math.PI / 2;
  shadowDisc.position.y = 0.015;
  shadowDisc.scale.setScalar(Math.max(0.78, heightScale));
  root.add(shadowDisc);

  // Guard bubble
  const guardGeo = new THREE.IcosahedronGeometry(1.24 * Math.max(0.8, heightScale), 1);
  const guardMat = new THREE.MeshBasicMaterial({
    color: weapon.accentHex,
    transparent: true,
    opacity: 0.22,
    wireframe: true,
    depthWrite: false,
  });
  geometries.push(guardGeo);
  materials.push(guardMat);
  const guardBubble = new THREE.Mesh(guardGeo, guardMat);
  guardBubble.position.y = 1.05 * heightScale;
  guardBubble.visible = false;
  root.add(guardBubble);

  // Glowing slash crescent shown dynamically during attack strikes
  const slashGeo = new THREE.RingGeometry(
    0.75 * heightScale,
    1.48 * heightScale,
    24,
    1,
    -Math.PI * 0.42,
    Math.PI * 0.84,
  );
  slashGeo.rotateX(-Math.PI / 2);
  slashGeo.rotateY(Math.PI / 2);
  const slashMat = new THREE.MeshBasicMaterial({
    color: weapon.accentHex,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  geometries.push(slashGeo);
  materials.push(slashMat);
  const attackSlashMesh = new THREE.Mesh(slashGeo, slashMat);
  attackSlashMesh.position.set(0, 1.15 * heightScale, 0.35 * heightScale);
  attackSlashMesh.visible = false;
  bodyGroup.add(attackSlashMesh);

  // Normalize scale and ground bottom-center flush at y=0
  const wrapper = new THREE.Group();
  const yawGroup = new THREE.Group();
  yawGroup.add(rawModel);
  wrapper.add(yawGroup);
  yawGroup.rotation.y = THREE.MathUtils.degToRad(asset.yawOffsetDeg || 0);
  wrapper.updateMatrixWorld(true);

  let box = computeBodyBox(false);
  if (box.isEmpty()) {
    box = computeBodyBox(true);
  }
  if (box.isEmpty()) {
    box = new THREE.Box3().setFromObject(wrapper);
  }
  const size = new THREE.Vector3();
  box.getSize(size);

  const isUprightCharacter = size.y >= Math.max(size.x, size.z) * 0.45;
  const refHeight = isUprightCharacter
    ? Math.max(size.y, 0.001)
    : Math.max(size.y, size.x * 0.75, size.z * 0.75, 0.001);
  const scale = targetHeight / refHeight;
  yawGroup.scale.multiplyScalar(scale);
  wrapper.updateMatrixWorld(true);

  let scaledBox = computeBodyBox(false);
  if (scaledBox.isEmpty()) {
    scaledBox = computeBodyBox(true);
  }
  if (scaledBox.isEmpty()) {
    scaledBox = new THREE.Box3().setFromObject(wrapper);
  }
  const scaledCenter = new THREE.Vector3();
  scaledBox.getCenter(scaledCenter);

  yawGroup.position.x -= scaledCenter.x;
  yawGroup.position.z -= scaledCenter.z;
  yawGroup.position.y -= scaledBox.min.y;
  wrapper.updateMatrixWorld(true);

  bodyGroup.add(wrapper);

  // Bind skeleton bones after scale/orientation alignment; fall back to part articulation for multi-part unrigged models (e.g. Wolf Full.dae)
  let bones = bindSkeletonBones(rawModel, wrapper);
  let hasSkeleton = Boolean(bones.shoulderR || bones.spineLower);
  let isPartArticulated = false;
  let partHandRCenterLocal: THREE.Vector3 | undefined;
  let partHandLCenterLocal: THREE.Vector3 | undefined;
  const backMissiles: THREE.Object3D[] = [];

  if (!hasSkeleton) {
    const articulated = tryArticulatePartRig(rawModel, wrapper, geometries, customCharacterId);
    if (articulated) {
      bodyGroup.add(articulated.articulatedRoot);
      bones = articulated.bones;
      hasSkeleton = Boolean(bones.shoulderR && bones.hipR);
      isPartArticulated = hasSkeleton;
      partHandRCenterLocal = articulated.handRCenterLocal;
      partHandLCenterLocal = articulated.handLCenterLocal;
      backMissiles.push(...articulated.backMissiles);
    }
  } else {
    rawModel.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.isMesh && getMeshPartTag(mesh) === 'missile') {
        backMissiles.push(mesh);
      }
    });
  }

  // Apply weapon separation / Hangar weapon attachment linked to Weapons tab
  const requestedMode = asset.weaponMode ?? 'auto';
  const signatureWeaponByChar: Record<ModelStyle, string> = {
    achilles: 'weapon_lance',
    hunter: 'weapon_rifle',
    kunoichi: 'weapon_kodachi',
    emperor: 'weapon_hammer',
    odin: 'weapon_halberd',
  };
  const matchesSignature = signatureWeaponByChar[customCharacterId] === loadout.weaponId;
  const hasCustomWeapon = Boolean(asset.customWeapon);
  const effectiveWeaponMode =
    requestedMode === 'detached'
      ? 'detached'
      : hasCustomWeapon && (requestedMode === 'auto' || !hasBuiltInWeapon)
        ? 'custom'
        : requestedMode === 'auto'
          ? customCharacterId === 'achilles'
            ? loadout.weaponId === 'weapon_lance'
              ? 'builtin'
              : 'hangar'
            : matchesSignature && hasBuiltInWeapon
              ? 'builtin'
              : 'hangar'
          : requestedMode;

  if (effectiveWeaponMode !== 'builtin') {
    weaponNodes.forEach((n) => {
      n.visible = false;
    });
  } else if (customCharacterId === 'emperor') {
    weaponNodes.forEach((n) => {
      let cur: THREE.Object3D | null = n.parent;
      let nested = false;
      while (cur && cur !== rawModel) {
        if (weaponNodeSet.has(cur)) {
          nested = true;
          break;
        }
        cur = cur.parent;
      }
      if (!nested) {
        n.scale.multiplyScalar(1.2);
      }
    });
  }

  let muzzlePoint = new THREE.Object3D();
  muzzlePoint.position.set(-0.28 * heightScale, 1.25 * heightScale, 0.9 * heightScale);
  bodyGroup.add(muzzlePoint);

  let bladeTipPoint = new THREE.Object3D();
  if (effectiveWeaponMode === 'builtin' && (bones.weaponR || bones.handR)) {
    const attachBone = bones.weaponR?.node ?? bones.handR!.node;
    wrapper.updateMatrixWorld(true);
    const boneWorld = attachBone.getWorldPosition(new THREE.Vector3());
    const handWorld = (bones.handR?.node ?? attachBone).getWorldPosition(new THREE.Vector3());
    let bestTipWorld: THREE.Vector3 | null = null;
    let bestDistSq = 0.06 * heightScale * heightScale;

    // 1. Check separate right-hand weapon mesh nodes (ignore shield nodes on +X left side)
    const tmpBox = new THREE.Box3();
    const tmpCenter = new THREE.Vector3();
    const tmpV = new THREE.Vector3();
    weaponNodes.forEach((wn) => {
      if (/shield|khien|buckler|aegis/i.test(wn.name)) return;
      tmpBox.setFromObject(wn);
      if (tmpBox.isEmpty()) return;
      tmpBox.getCenter(tmpCenter);
      bodyGroup.worldToLocal(tmpCenter);
      if (tmpCenter.x > 0.18 * heightScale) return; // left-side shield
      wn.traverse((ch) => {
        const m = ch as THREE.Mesh;
        if (!m.isMesh || !m.geometry) return;
        const pos = m.geometry.getAttribute('position');
        if (!pos) return;
        for (let i = 0; i < pos.count; i++) {
          tmpV.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
          const dSq = tmpV.distanceToSquared(handWorld);
          if (dSq > bestDistSq) {
            bestDistSq = dSq;
            bestTipWorld = tmpV.clone();
          }
        }
      });
    });

    // 2. Also scan SkinnedMesh vertices weighted to weaponR / handR (handles single-mesh FBX/GLB models)
    if (!bestTipWorld) {
      rawModel.traverse((ch) => {
        const sm = ch as THREE.SkinnedMesh;
        if (!sm.isSkinnedMesh || !sm.skeleton || !sm.geometry) return;
        const pos = sm.geometry.getAttribute('position');
        const skinIdx = sm.geometry.getAttribute('skinIndex');
        const skinWgt = sm.geometry.getAttribute('skinWeight');
        if (!pos || !skinIdx || !skinWgt) return;
        const targetBoneIndices = new Set<number>();
        sm.skeleton.bones.forEach((b, idx) => {
          if (b === bones.weaponR?.node || b === bones.handR?.node) {
            targetBoneIndices.add(idx);
          }
        });
        if (targetBoneIndices.size === 0) return;
        for (let i = 0; i < pos.count; i++) {
          let wSum = 0;
          if (targetBoneIndices.has(skinIdx.getX(i))) wSum += skinWgt.getX(i);
          if (targetBoneIndices.has(skinIdx.getY(i))) wSum += skinWgt.getY(i);
          if (targetBoneIndices.has(skinIdx.getZ(i))) wSum += skinWgt.getZ(i);
          if (targetBoneIndices.has(skinIdx.getW(i))) wSum += skinWgt.getW(i);
          if (wSum < 0.35) continue;
          tmpV.fromBufferAttribute(pos, i).applyMatrix4(sm.matrixWorld);
          const dSq = tmpV.distanceToSquared(handWorld);
          if (dSq > bestDistSq) {
            bestDistSq = dSq;
            bestTipWorld = tmpV.clone();
          }
        }
      });
    }

    if (bestTipWorld) {
      attachBone.worldToLocal(bestTipWorld);
      bladeTipPoint.position.copy(bestTipWorld);
    } else {
      const fallbackWorld = boneWorld.clone().add(new THREE.Vector3(0, 0.25 * heightScale, 0.85 * heightScale));
      attachBone.worldToLocal(fallbackWorld);
      bladeTipPoint.position.copy(fallbackWorld);
    }
    attachBone.add(bladeTipPoint);
  } else {
    bladeTipPoint.position.set(-0.28 * heightScale, 1.2 * heightScale, 1.25 * heightScale);
    bodyGroup.add(bladeTipPoint);
  }

  const measureAxisEndCrossSection = (
    obj: THREE.Object3D,
    axis: 'x' | 'z',
    isMinEnd: boolean,
    bounds: THREE.Box3,
  ): number => {
    const span = axis === 'x' ? bounds.max.x - bounds.min.x : bounds.max.z - bounds.min.z;
    const cutoff = isMinEnd
      ? (axis === 'x' ? bounds.min.x : bounds.min.z) + span * 0.2
      : (axis === 'x' ? bounds.max.x : bounds.max.z) - span * 0.2;
    const sliceBox = new THREE.Box3();
    const v = new THREE.Vector3();
    obj.updateMatrixWorld(true);
    obj.traverse((ch) => {
      const m = ch as THREE.Mesh;
      if (!m.isMesh || !m.geometry) return;
      const pos = m.geometry.getAttribute('position');
      if (!pos) return;
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
        const coord = axis === 'x' ? v.x : v.z;
        if ((isMinEnd && coord <= cutoff) || (!isMinEnd && coord >= cutoff)) {
          sliceBox.expandByPoint(v);
        }
      }
    });
    if (sliceBox.isEmpty()) return 0;
    const s = sliceBox.getSize(new THREE.Vector3());
    return s.y + (axis === 'x' ? s.z : s.x);
  };

  let hasHangarWeapon = false;
  if (effectiveWeaponMode === 'custom' && asset.customWeapon) {
    try {
      const parsedWep = await parseModelObject(asset.customWeapon);
      const rawWep = parsedWep.object;
      extraDispose = parsedWep.disposeResources;

      rawWep.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (mesh.isMesh) {
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          if (mesh.geometry) geometries.push(mesh.geometry);
          const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          mats.forEach((m) => {
            if (!m) return;
            materials.push(m);
            const isOutlineMat = /^line(_?m)?$/i.test(m.name || '');
            if ((m as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
              const std = m as THREE.MeshStandardMaterial;
              std.envMapIntensity = Math.max(std.envMapIntensity ?? 1, 0.85);
              if (!isOutlineMat) hitFlashMaterials.push(std);
            } else if (
              !isOutlineMat &&
              ((m as THREE.MeshPhongMaterial).isMeshPhongMaterial ||
                (m as THREE.MeshLambertMaterial).isMeshLambertMaterial)
            ) {
              hitFlashMaterials.push(m as unknown as THREE.MeshStandardMaterial);
            }
          });
        }
      });

      const wWrapper = new THREE.Group();
      const wYawGroup = new THREE.Group();
      const wAutoAlign = new THREE.Group();
      wAutoAlign.add(rawWep);
      wYawGroup.add(wAutoAlign);
      wWrapper.add(wYawGroup);

      // Stand lying-down weapons upright (+Y up) and align barrel/weapon head straight forward (+Z)
      wAutoAlign.updateMatrixWorld(true);
      const rawWBox = new THREE.Box3().setFromObject(wAutoAlign);
      const rawWSize = rawWBox.getSize(new THREE.Vector3());
      wAutoAlign.rotation.order = 'YXZ';
      if (rawWSize.y < rawWSize.x * 0.55 && rawWSize.y < rawWSize.z * 0.55) {
        wAutoAlign.rotation.x = -Math.PI / 2;
      }
      wAutoAlign.updateMatrixWorld(true);

      const isHammerWeapon = loadout.weaponId === 'weapon_hammer';
      const uprightBox = new THREE.Box3().setFromObject(wAutoAlign);
      const uprightSize = uprightBox.getSize(new THREE.Vector3());
      if (uprightSize.x > uprightSize.z * 1.35) {
        const minCross = measureAxisEndCrossSection(wAutoAlign, 'x', true, uprightBox);
        const maxCross = measureAxisEndCrossSection(wAutoAlign, 'x', false, uprightBox);
        if (isHammerWeapon) {
          // For heavy hammer/axe, the wider heavy head points forward (+Z)
          wAutoAlign.rotation.y = minCross >= maxCross ? Math.PI / 2 : -Math.PI / 2;
        } else {
          // Narrower end is barrel tip; point it along +Z
          wAutoAlign.rotation.y = minCross <= maxCross ? Math.PI / 2 : -Math.PI / 2;
        }
      } else if (uprightSize.z > uprightSize.x * 1.35) {
        const minCross = measureAxisEndCrossSection(wAutoAlign, 'z', true, uprightBox);
        const maxCross = measureAxisEndCrossSection(wAutoAlign, 'z', false, uprightBox);
        if (isHammerWeapon) {
          if (minCross > maxCross * 1.15) {
            wAutoAlign.rotation.y = Math.PI;
          }
        } else if (minCross < maxCross * 0.85) {
          wAutoAlign.rotation.y = Math.PI;
        }
      }

      wYawGroup.rotation.y = THREE.MathUtils.degToRad(asset.customWeapon.yawOffsetDeg || 0);
      wWrapper.updateMatrixWorld(true);

      const wBox = new THREE.Box3().setFromObject(wWrapper);
      const wSize = new THREE.Vector3();
      wBox.getSize(wSize);
      const targetLen =
        (loadout.weaponId === 'weapon_rifle'
          ? 1.62
          : isHammerWeapon
            ? 2.24
            : 1.54) * heightScale;
      const wScale = targetLen / Math.max(wSize.x, wSize.y, wSize.z, 0.001);
      wYawGroup.scale.multiplyScalar(wScale);
      wWrapper.updateMatrixWorld(true);

      const wsBox = new THREE.Box3().setFromObject(wWrapper);
      const wsCenter = new THREE.Vector3();
      const wsSize = new THREE.Vector3();
      wsBox.getCenter(wsCenter);
      wsBox.getSize(wsSize);

      wYawGroup.position.x -= wsCenter.x;
      wYawGroup.position.y -=
        wsBox.min.y +
        wsSize.y * (loadout.weaponId === 'weapon_rifle' ? 0.36 : isHammerWeapon ? 0.5 : 0.48);
      wYawGroup.position.z -= wsBox.min.z + wsSize.z * (isHammerWeapon ? 0.31 : 0.28);
      wWrapper.updateMatrixWorld(true);

      const finalWepBox = new THREE.Box3().setFromObject(wWrapper);
      const finalWepSize = finalWepBox.getSize(new THREE.Vector3());

      hasHangarWeapon = true;
      if (bones.handR) {
        const handWorldPos = bones.handR.node.getWorldPosition(new THREE.Vector3());
        const handWorldQuat = bones.handR.node.getWorldQuaternion(new THREE.Quaternion());
        const handWorldScale = bones.handR.node.getWorldScale(new THREE.Vector3());
        const attachR = new THREE.Group();
        if (isPartArticulated && partHandRCenterLocal) {
          attachR.position.copy(partHandRCenterLocal);
        } else {
          const palmTarget = handWorldPos.clone().add(new THREE.Vector3(0, -0.06, 0.04));
          bones.handR.node.worldToLocal(palmTarget);
          attachR.position.copy(palmTarget);
        }
        attachR.quaternion.copy(handWorldQuat).invert();
        attachR.scale.set(
          1 / Math.max(0.0001, handWorldScale.x),
          1 / Math.max(0.0001, handWorldScale.y),
          1 / Math.max(0.0001, handWorldScale.z),
        );
        attachR.add(wWrapper);
        bones.handR.node.add(attachR);
      } else {
        const sideX = Math.min(-0.42 * heightScale, scaledBox.min.x - 0.08 * heightScale);
        const sideY = Math.max(0.95 * heightScale, scaledBox.max.y * 0.56);
        wWrapper.position.set(sideX, sideY, 0.22 * heightScale);
        bodyGroup.add(wWrapper);
      }

      const customMuzzle = new THREE.Object3D();
      customMuzzle.position.set(0, finalWepBox.min.y + finalWepSize.y * 0.44, finalWepBox.max.z);
      wWrapper.add(customMuzzle);
      muzzlePoint = customMuzzle;

      const customBladeTip = new THREE.Object3D();
      customBladeTip.position.set(
        0,
        isHammerWeapon ? 0 : finalWepBox.min.y + finalWepSize.y * 0.44,
        isHammerWeapon ? finalWepBox.max.z * 0.88 : finalWepBox.max.z,
      );
      wWrapper.add(customBladeTip);
      bladeTipPoint = customBladeTip;
    } catch {
      // Fallback safely if weapon model fails to parse
    }
  } else if (effectiveWeaponMode === 'hangar') {
    hasHangarWeapon = true;
    const hangarProps = buildHangarWeaponGroups(loadout);
    extraDispose = hangarProps.dispose;

    if (bones.handR) {
      const handWorldPos = bones.handR.node.getWorldPosition(new THREE.Vector3());
      const handWorldQuat = bones.handR.node.getWorldQuaternion(new THREE.Quaternion());
      const handWorldScale = bones.handR.node.getWorldScale(new THREE.Vector3());
      const isRifle = loadout.weaponId === 'weapon_rifle';
      const isHammer = loadout.weaponId === 'weapon_hammer';
      const propScale = (isRifle ? 0.86 : isHammer ? 1.15 : 0.94) * heightScale;
      const attachR = new THREE.Group();
      if (customCharacterId === 'achilles') {
        attachR.position.set(
          isRifle ? -0.14 : -0.11,
          isRifle ? 0.34 : 0.28,
          isRifle ? 0.08 : 0.12,
        );
      } else if (isPartArticulated && partHandRCenterLocal) {
        attachR.position.copy(partHandRCenterLocal);
      } else {
        const palmTarget = handWorldPos.clone().add(new THREE.Vector3(0, -0.1 * heightScale, 0.03 * heightScale));
        bones.handR.node.worldToLocal(palmTarget);
        attachR.position.copy(palmTarget);
      }
      attachR.quaternion.copy(handWorldQuat).invert();
      attachR.scale.set(
        propScale / Math.max(0.0001, handWorldScale.x),
        propScale / Math.max(0.0001, handWorldScale.y),
        propScale / Math.max(0.0001, handWorldScale.z),
      );
      if (isRifle) {
        hangarProps.weaponGroup.position.set(0, 0.04, 0.07);
      }
      attachR.add(hangarProps.weaponGroup);
      bones.handR.node.add(attachR);
    } else {
      const sideX = Math.min(-0.42 * heightScale, scaledBox.min.x - 0.08 * heightScale);
      const sideY = Math.max(0.95 * heightScale, scaledBox.max.y * 0.56);
      hangarProps.weaponGroup.scale.setScalar(heightScale);
      hangarProps.weaponGroup.position.set(sideX, sideY, 0.22 * heightScale);
      bodyGroup.add(hangarProps.weaponGroup);
    }

    if (bones.handL) {
      const handLPos = bones.handL.node.getWorldPosition(new THREE.Vector3());
      const handLQuat = bones.handL.node.getWorldQuaternion(new THREE.Quaternion());
      const handLScale = bones.handL.node.getWorldScale(new THREE.Vector3());
      const attachL = new THREE.Group();
      if (customCharacterId === 'achilles') {
        attachL.position.set(0.16, 0.24, 0.04);
      } else if (isPartArticulated && partHandLCenterLocal) {
        attachL.position.copy(partHandLCenterLocal);
      } else {
        const palmTargetL = handLPos.clone().add(new THREE.Vector3(0, -0.1 * heightScale, 0.03 * heightScale));
        bones.handL.node.worldToLocal(palmTargetL);
        attachL.position.copy(palmTargetL);
      }
      attachL.quaternion.copy(handLQuat).invert();
      attachL.scale.set(
        heightScale / Math.max(0.0001, handLScale.x),
        heightScale / Math.max(0.0001, handLScale.y),
        heightScale / Math.max(0.0001, handLScale.z),
      );
      attachL.add(hangarProps.shieldGroup);
      bones.handL.node.add(attachL);
    } else {
      const sideL = Math.max(0.45 * heightScale, scaledBox.max.x + 0.06 * heightScale);
      const sideLY = Math.max(0.98 * heightScale, scaledBox.max.y * 0.56);
      hangarProps.shieldGroup.scale.setScalar(heightScale);
      hangarProps.shieldGroup.position.set(sideL, sideLY, 0.15 * heightScale);
      bodyGroup.add(hangarProps.shieldGroup);
    }

    muzzlePoint = hangarProps.muzzlePoint;
    bladeTipPoint = hangarProps.bladeTipPoint;
  }

  // Optional embedded animations (disabled by default, also pre-bound for special attack playback)
  let mixer: THREE.AnimationMixer | undefined;
  let specialMixer: THREE.AnimationMixer | undefined;
  let specialAction: THREE.AnimationAction | undefined;
  let specialClipDuration: number | undefined;
  let poseSnapshot:
    | Array<{
        node: THREE.Object3D;
        pos: THREE.Vector3;
        quat: THREE.Quaternion;
        scale: THREE.Vector3;
      }>
    | undefined;

  if (animations.length > 0) {
    if (asset.playAnimation) {
      mixer = new THREE.AnimationMixer(rawModel);
      const clip = animations[0];
      if (clip) {
        mixer.clipAction(clip).play();
      }
    } else {
      const snap: NonNullable<CustomLbxRig['poseSnapshot']> = [];
      rawModel.traverse((node) => {
        snap.push({
          node,
          pos: node.position.clone(),
          quat: node.quaternion.clone(),
          scale: node.scale.clone(),
        });
      });
      poseSnapshot = snap;
      const specialClip =
        animations.find((c) => /attack|strike|special|skill|slash|action|thunder/i.test(c.name)) ??
        animations[0];
      if (specialClip) {
        specialMixer = new THREE.AnimationMixer(rawModel);
        specialAction = specialMixer.clipAction(specialClip);
        specialClipDuration = specialClip.duration;
      }
    }
  }

  // Record Achilles' posed crown height directly so subsequent Hunter/Kunoichi builds reuse it instantly
  if (customCharacterId === 'achilles') {
    if (mixer) {
      mixer.update(0.15);
    } else if (hasSkeleton) {
      const ready = buildCustomLancePose({ mode: 'ready', elapsed: 0 });
      if (bones.spineLower) {
        rotateBoneInCharSpace(bones.spineLower, ready.torsoPitch * 0.45, ready.torsoYaw * 0.55, ready.torsoRoll * 0.45);
        rotateBoneInCharSpace(bones.spineUpper, ready.torsoPitch * 0.55, ready.torsoYaw * 0.45, ready.torsoRoll * 0.55);
      } else {
        rotateBoneInCharSpace(bones.spineUpper, ready.torsoPitch, ready.torsoYaw, ready.torsoRoll);
      }
      rotateBoneInCharSpace(bones.head, ready.headPitch, ready.headYaw, 0);
      rotateBoneInCharSpace(bones.hipL, ready.lHipX, 0, ready.lHipZ);
      rotateBoneInCharSpace(bones.kneeL, ready.lKneeX, 0, 0);
      rotateBoneInCharSpace(bones.hipR, ready.rHipX, 0, ready.rHipZ);
      rotateBoneInCharSpace(bones.kneeR, ready.rKneeX, 0, 0);
    }
    cachedAchillesCrown = {
      key: achillesCacheKey(asset),
      height: estimateAchillesCrownHeightFromPosed(rawModel, wrapper, bones),
    };
    if (mixer) {
      mixer.setTime(0);
    } else if (hasSkeleton) {
      rotateBoneInCharSpace(bones.spineLower, 0, 0, 0);
      rotateBoneInCharSpace(bones.spineUpper, 0, 0, 0);
      rotateBoneInCharSpace(bones.head, 0, 0, 0);
      rotateBoneInCharSpace(bones.hipL, 0, 0, 0);
      rotateBoneInCharSpace(bones.kneeL, 0, 0, 0);
      rotateBoneInCharSpace(bones.hipR, 0, 0, 0);
      rotateBoneInCharSpace(bones.kneeR, 0, 0, 0);
    }
    wrapper.updateMatrixWorld(true);
  }

  // Standard dummy pivots for LbxRig interface compatibility
  const pelvis = new THREE.Group();
  const torsoPivot = new THREE.Group();
  const headPivot = new THREE.Group();
  const leftShoulder = new THREE.Group();
  const leftElbow = new THREE.Group();
  const rightShoulder = new THREE.Group();
  const rightElbow = new THREE.Group();
  const leftHip = new THREE.Group();
  const leftKnee = new THREE.Group();
  const rightHip = new THREE.Group();
  const rightKnee = new THREE.Group();
  const boosterPivot = new THREE.Group();
  const weaponPivot = new THREE.Group();
  const shieldPivot = new THREE.Group();

  const dispose = () => {
    mixer?.uncacheRoot(rawModel);
    specialMixer?.uncacheRoot(rawModel);
    disposeResources();
    extraDispose?.();
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => m.dispose());
    textures.forEach((t) => t.dispose());
  };

  return {
    root,
    bodyGroup,
    pelvis,
    torsoPivot,
    headPivot,
    leftShoulder,
    leftElbow,
    rightShoulder,
    rightElbow,
    leftHip,
    leftKnee,
    rightHip,
    rightKnee,
    boosterPivot,
    weaponPivot,
    shieldPivot,
    muzzlePoint,
    bladeTipPoint,
    thrusterFlames: [],
    shadowDisc,
    guardBubble,
    hitFlashMaterials,
    mixer,
    specialMixer,
    specialAction,
    specialClipDuration,
    specialAnimActive: false,
    poseSnapshot,
    backMissiles,
    isCustomModel: true,
    customCharacterId,
    animationCount: animations.length,
    hasSkeleton,
    isPartArticulated,
    hasBuiltInWeapon,
    hasHangarWeapon,
    bones,
    poseState: createZeroCustomPose(),
    attackSlashMesh,
    dispose,
  };
}
