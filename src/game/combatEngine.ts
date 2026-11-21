import * as THREE from 'three';
import { TacticalAiBrain } from './aiController';
import {
  buildArenaScene,
  detectSoftwareWebGL,
  InteractableRuntimeVisual,
  ObstacleRuntimeVisual,
  setupStudioEnvironment,
} from './arenas';
import { computeLoadoutStats, getArenaById, getWeaponById } from './catalog';
import {
  applyCustomBonePose,
  buildCustomJumpPose,
  buildCustomLancePose,
  buildCustomLbxRig,
  buildCustomMeleePose,
  buildCustomRiflePose,
  createZeroCustomPose,
  CustomBonePose,
  CustomLbxRig,
  startCustomSpecialClip,
  stopCustomSpecialClip,
} from './customModelLoader';
import { CustomModelAsset } from './customModelStorage';
import { buildLbxModel } from './lbxModel';
import { soundEngine } from './sound';
import {
  AiPersonality,
  AiTacticalState,
  ArenaConfig,
  ArenaId,
  ArenaInteractable,
  ArenaObstacle,
  BattleHudSnapshot,
  ComputedLbxStats,
  LbxLoadout,
  LbxWeapon,
} from './types';

interface FighterRuntime {
  isPlayer: boolean;
  name: string;
  loadout: LbxLoadout;
  stats: ComputedLbxStats;
  weapon: LbxWeapon;
  rig: CustomLbxRig;

  pos: THREE.Vector3;
  vel: THREE.Vector3;
  velY: number;
  groundY: number;
  isGrounded: boolean;
  jumpSquashTimer: number;
  airAttacking: boolean;
  yaw: number;
  targetYaw: number;

  hp: number;
  energy: number;
  special: number;

  isGuarding: boolean;
  isAiming: boolean;
  guardStartTime: number;
  isDashing: boolean;
  dashTimer: number;
  dashDir: THREE.Vector3;

  attackPhase: 'none' | 'windup' | 'active' | 'recover';
  attackTimer: number;
  attackDuration: number;
  comboStep: number;
  comboResetTimer: number;
  attackCooldown: number;
  hasHitCurrentSwing: boolean;
  isSpecialAttack: boolean;
  specialStage: number;
  missileRearmTimer: number;
  launchedBySpecial: boolean;

  stunTimer: number;
  hitFlashTimer: number;
  invulnTimer: number;
  comboCounter: number;
  comboDecayTimer: number;
  stridePhase: number;
  aimTorsoYaw: number;
  aimTorsoPitch: number;
  aiState?: AiTacticalState;
}

interface ObstacleState {
  data: ArenaObstacle;
  id: string;
  hp: number;
  destroyed: boolean;
}

interface InteractableState {
  data: ArenaInteractable;
  cooldown: number;
}

interface Projectile {
  active: boolean;
  ownerIsPlayer: boolean;
  pos: THREE.Vector3;
  prevPos: THREE.Vector3;
  vel: THREE.Vector3;
  damage: number;
  life: number;
  isSpecial: boolean;
  mesh: THREE.Group;
}

interface StingerMissile {
  active: boolean;
  ownerIsPlayer: boolean;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  age: number;
  life: number;
  damage: number;
  smokeTimer: number;
  isFinal: boolean;
  mesh: THREE.Group;
  defaultVisual: THREE.Group;
  customVisual: THREE.Object3D | null;
}

interface LightningBolt {
  active: boolean;
  life: number;
  maxLife: number;
  group: THREE.Group;
  segments: THREE.Mesh[];
  mats: THREE.MeshBasicMaterial[];
}

interface SparkParticle {
  active: boolean;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  life: number;
  maxLife: number;
  mesh: THREE.Mesh;
}

interface ShockwaveRing {
  active: boolean;
  life: number;
  maxLife: number;
  maxRadius: number;
  mesh: THREE.Mesh;
}

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _zAxis = new THREE.Vector3(0, 0, 1);
const _camForward = new THREE.Vector3();
const _camRight = new THREE.Vector3();
const _desiredCamPos = new THREE.Vector3();
const _desiredLook = new THREE.Vector3();
const _currentCamLook = new THREE.Vector3();

function shortestAngleDiff(from: number, to: number): number {
  let diff = (to - from) % (Math.PI * 2);
  if (diff > Math.PI) diff -= Math.PI * 2;
  if (diff < -Math.PI) diff += Math.PI * 2;
  return diff;
}

function dampAngle(current: number, target: number, lambda: number, dt: number): number {
  const diff = shortestAngleDiff(current, target);
  return current + diff * (1 - Math.exp(-lambda * dt));
}

export interface CombatCallbacks {
  onHudUpdate: (snapshot: BattleHudSnapshot) => void;
  onBattleEnd: (outcome: 'victory' | 'defeat', snapshot: BattleHudSnapshot) => void;
  onPointerLockChange?: (locked: boolean) => void;
  onRequestPause?: () => void;
}

export class LbxCombatEngine {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private isSoftwareGl: boolean;
  private disposeEnv: () => void;
  private disposeArena: () => void;
  private sunLight: THREE.DirectionalLight | null = null;
  private readonly sunDir = new THREE.Vector3(34, 44, 26).normalize();

  private arena: ArenaConfig;
  private obstacleStates: ObstacleState[];
  private obstacleVisuals: Map<string, ObstacleRuntimeVisual>;
  private interactableStates: InteractableState[];
  private interactableVisuals: Map<string, InteractableRuntimeVisual>;
  private activeObstaclesScratch: ArenaObstacle[] = [];
  private readyInteractablesScratch: ArenaInteractable[] = [];

  private player: FighterRuntime;
  private enemy: FighterRuntime;
  private aiBrain: TacticalAiBrain;
  private callbacks: CombatCallbacks;

  private keys = new Set<string>();
  private mouseAttackQueued = false;
  private mouseGuardHeld = false;
  private lockOnActive = false;
  private camOrbitYaw = Math.PI;
  private camOrbitPitch = 0.36;
  private camDistance = 7.0;
  private cameraShake = 0;
  private lastPointerX: number | null = null;
  private lastPointerY: number | null = null;
  private pointerEdgeRate = 0;
  private activeTouchId: number | null = null;
  private lastTouchX = 0;
  private lastTouchY = 0;
  private mouseSensitivity = 1.0;
  private wasPointerLocked = false;

  private projectiles: Projectile[] = [];
  private missiles: StingerMissile[] = [];
  private lightningBolts: LightningBolt[] = [];
  private sparks: SparkParticle[] = [];
  private shockwaves: ShockwaveRing[] = [];
  private thunderLight!: THREE.PointLight;
  private thunderAuraGroup!: THREE.Group;
  private thunderOuterRing!: THREE.Mesh;
  private thunderInnerRing!: THREE.Mesh;
  private thunderPillar!: THREE.Mesh;
  private thunderSpearOrb!: THREE.Mesh;
  private thunderSpearGlow!: THREE.Mesh;
  private thunderSpearArcA!: THREE.Mesh;
  private thunderSpearArcB!: THREE.Mesh;
  private thunderCurtainStreaks: THREE.Mesh[] = [];
  private thunderLanceCone!: THREE.Mesh;
  private thunderConeRings: THREE.Mesh[] = [];
  private thunderAuraTimer = 0;
  private thunderFlashIntensity = 0;
  private thunderCutInProgress = -1;
  private lastThunderShot = -1;
  private thunderShotActive = false;
  private thunderShotTimer = 0;
  private thunderShotDuration = 0.34;
  private thunderShotOrigin = new THREE.Vector3();
  private thunderShotPos = new THREE.Vector3();
  private thunderShotCaster: FighterRuntime | null = null;
  private thunderShotTarget: FighterRuntime | null = null;
  private thunderShotArcCooldown = 0;
  private sharedGeos: THREE.BufferGeometry[] = [];
  private sharedMats: THREE.Material[] = [];

  private clock = new THREE.Clock();
  private elapsedTime = 0;
  private hudEmitTimer = 0;
  private bannerText: string | null = 'БОЙ НАЧАЛСЯ!';
  private bannerTimer = 1.8;
  private outcome: 'ongoing' | 'victory' | 'defeat' = 'ongoing';
  private outcomeDelayTimer = 0;
  private paused = false;

  private damageDealt = 0;
  private parriesLanded = 0;
  private specialsUsed = 0;

  private frameId = 0;
  private disposed = false;
  private virtualMove = { x: 0, z: 0 };

  constructor(
    container: HTMLElement,
    playerLoadout: LbxLoadout,
    enemyLoadout: LbxLoadout,
    enemyName: string,
    arenaId: ArenaId,
    aiPersonality: AiPersonality,
    callbacks: CombatCallbacks,
    playerCustomModel?: CustomModelAsset | null,
    mouseSensitivity = 1.0,
    enemyCustomModel?: CustomModelAsset | null,
  ) {
    this.container = container;
    this.callbacks = callbacks;
    this.mouseSensitivity = THREE.MathUtils.clamp(mouseSensitivity, 0.2, 2.5);
    this.isSoftwareGl = detectSoftwareWebGL();
    this.arena = getArenaById(arenaId);

    this.renderer = new THREE.WebGLRenderer({
      antialias: !this.isSoftwareGl,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.setPixelRatio(
      this.isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2),
    );
    this.renderer.shadowMap.enabled = !this.isSoftwareGl;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.arena.skyColor);
    this.scene.fog = new THREE.FogExp2(
      this.arena.fogColor,
      this.arena.id === 'meadow' ? 0.0024 : 0.011,
    );

    this.camera = new THREE.PerspectiveCamera(
      48,
      Math.max(1, container.clientWidth) / Math.max(1, container.clientHeight),
      0.1,
      520,
    );

    this.disposeEnv = setupStudioEnvironment(
      this.renderer,
      this.scene,
      this.isSoftwareGl,
      this.arena.id === 'meadow',
      (light) => {
        this.sunLight = light;
      },
    );

    const builtArena = buildArenaScene(
      this.arena,
      this.renderer.capabilities.getMaxAnisotropy(),
    );
    this.scene.add(builtArena.group);
    this.disposeArena = builtArena.dispose;
    this.obstacleVisuals = builtArena.obstacleVisuals;
    this.interactableVisuals = builtArena.interactableVisuals;

    this.obstacleStates = this.arena.obstacles.map((o, idx) => ({
      data: o,
      id: o.id ?? `obs_${idx}`,
      hp: o.maxHp ?? 140,
      destroyed: false,
    }));
    this.interactableStates = this.arena.interactables.map((node) => ({
      data: node,
      cooldown: 0,
    }));

    const spawnOffset =
      this.arena.id === 'meadow' ? 21.0 : Math.min(13.5, this.arena.radius * 0.56);
    this.player = this.createFighter(true, 'Игрок (Бан)', playerLoadout, -spawnOffset);
    this.enemy = this.createFighter(false, enemyName, enemyLoadout, spawnOffset);
    this.player.yaw = 0;
    this.player.targetYaw = 0;
    this.enemy.yaw = Math.PI;
    this.enemy.targetYaw = Math.PI;
    this.player.rig.root.rotation.y = this.player.yaw;
    this.enemy.rig.root.rotation.y = this.enemy.yaw;

    this.scene.add(this.player.rig.root);
    this.scene.add(this.enemy.rig.root);

    if (playerCustomModel) {
      void buildCustomLbxRig(playerCustomModel, playerLoadout)
        .then((customRig) => {
          if (this.disposed) {
            customRig.dispose();
            return;
          }
          const prevRig = this.player.rig;
          customRig.root.position.copy(this.player.pos);
          customRig.root.rotation.y = this.player.yaw;
          customRig.guardBubble.visible = prevRig.guardBubble.visible;
          this.scene.remove(prevRig.root);
          prevRig.dispose();
          this.player.rig = customRig;
          this.scene.add(customRig.root);
        })
        .catch(() => {
          // Keep procedural rig if custom model fails to parse
        });
    }

    if (enemyCustomModel) {
      void buildCustomLbxRig(enemyCustomModel, enemyLoadout)
        .then((customRig) => {
          if (this.disposed) {
            customRig.dispose();
            return;
          }
          const prevRig = this.enemy.rig;
          customRig.root.position.copy(this.enemy.pos);
          customRig.root.rotation.y = this.enemy.yaw;
          customRig.guardBubble.visible = prevRig.guardBubble.visible;
          this.scene.remove(prevRig.root);
          prevRig.dispose();
          this.enemy.rig = customRig;
          this.scene.add(customRig.root);
        })
        .catch(() => {
          // Keep procedural rig if custom model fails to parse
        });
    }

    this.aiBrain = new TacticalAiBrain(aiPersonality);

    this.initVfxPools();
    this.initCameraPosition();

    if (!this.isSoftwareGl) {
      this.renderer.compile(this.scene, this.camera);
    }

    this.bindEvents();
    this.updateCursorVisibility();
    this.requestPointerLock();
    this.emitHud();
    this.tick();
  }

  private createFighter(
    isPlayer: boolean,
    name: string,
    loadout: LbxLoadout,
    startZ: number,
  ): FighterRuntime {
    const stats = computeLoadoutStats(loadout);
    const weapon = getWeaponById(loadout.weaponId);
    const rig = buildLbxModel(loadout);
    rig.root.position.set(0, 0, startZ);

    return {
      isPlayer,
      name,
      loadout,
      stats,
      weapon,
      rig,
      pos: new THREE.Vector3(0, 0, startZ),
      vel: new THREE.Vector3(),
      velY: 0,
      groundY: 0,
      isGrounded: true,
      jumpSquashTimer: 0,
      airAttacking: false,
      yaw: 0,
      targetYaw: 0,
      hp: stats.maxHp,
      energy: stats.maxEnergy,
      special: 25,
      isGuarding: false,
      isAiming: false,
      guardStartTime: 0,
      isDashing: false,
      dashTimer: 0,
      dashDir: new THREE.Vector3(),
      attackPhase: 'none',
      attackTimer: 0,
      attackDuration: 0.4,
      comboStep: 0,
      comboResetTimer: 0,
      attackCooldown: 0,
      hasHitCurrentSwing: false,
      isSpecialAttack: false,
      specialStage: 0,
      missileRearmTimer: 0,
      launchedBySpecial: false,
      stunTimer: 0,
      hitFlashTimer: 0,
      invulnTimer: 0,
      comboCounter: 0,
      comboDecayTimer: 0,
      stridePhase: 0,
      aimTorsoYaw: 0,
      aimTorsoPitch: 0,
    };
  }

  private initVfxPools() {
    // Sleek futuristic plasma-rail sniper bolts (white-hot needle core + additive glowing sheath + nose flare)
    const coreGeo = new THREE.CylinderGeometry(0.028, 0.014, 1.35, 8);
    coreGeo.rotateX(Math.PI / 2);
    const sheathGeo = new THREE.CylinderGeometry(0.088, 0.036, 1.75, 8);
    sheathGeo.rotateX(Math.PI / 2);
    const tipGeo = new THREE.SphereGeometry(0.09, 8, 8);
    const finGeo = new THREE.BoxGeometry(0.24, 0.014, 0.55);
    this.sharedGeos.push(coreGeo, sheathGeo, tipGeo, finGeo);

    const coreMat = new THREE.MeshBasicMaterial({ color: '#FFFFFF' });
    const playerSheathMat = new THREE.MeshBasicMaterial({
      color: '#38BDF8',
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const enemySheathMat = new THREE.MeshBasicMaterial({
      color: '#FB923C',
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedMats.push(coreMat, playerSheathMat, enemySheathMat);

    for (let i = 0; i < 24; i++) {
      const isPlayer = i < 12;
      const sheathMat = isPlayer ? playerSheathMat : enemySheathMat;
      const grp = new THREE.Group();
      const core = new THREE.Mesh(coreGeo, coreMat);
      const sheath = new THREE.Mesh(sheathGeo, sheathMat);
      const tip = new THREE.Mesh(tipGeo, sheathMat);
      tip.position.z = 0.65;
      const fins = new THREE.Mesh(finGeo, sheathMat);
      fins.position.z = -0.35;
      grp.add(core, sheath, tip, fins);
      grp.visible = false;
      this.scene.add(grp);
      this.projectiles.push({
        active: false,
        ownerIsPlayer: isPlayer,
        pos: new THREE.Vector3(),
        prevPos: new THREE.Vector3(),
        vel: new THREE.Vector3(),
        damage: 78,
        life: 0,
        isSpecial: false,
        mesh: grp,
      });
    }

    // Stinger Homing Missile Pool (fallback sleek tactical missile + slot for cloned Hunter back-missiles)
    const mslBodyGeo = new THREE.CylinderGeometry(0.055, 0.06, 0.52, 8);
    mslBodyGeo.rotateX(Math.PI / 2);
    const mslConeGeo = new THREE.ConeGeometry(0.062, 0.18, 8);
    mslConeGeo.rotateX(Math.PI / 2);
    const mslFinGeo = new THREE.BoxGeometry(0.22, 0.015, 0.14);
    const mslFlameGeo = new THREE.ConeGeometry(0.055, 0.34, 8);
    mslFlameGeo.rotateX(-Math.PI / 2);
    this.sharedGeos.push(mslBodyGeo, mslConeGeo, mslFinGeo, mslFlameGeo);

    const mslBodyMat = new THREE.MeshStandardMaterial({
      color: '#CBD5E1',
      metalness: 0.72,
      roughness: 0.25,
    });
    const mslConeMat = new THREE.MeshStandardMaterial({
      color: '#EF4444',
      metalness: 0.4,
      roughness: 0.35,
    });
    const mslFlameMat = new THREE.MeshBasicMaterial({
      color: '#FDE047',
      transparent: true,
      opacity: 0.92,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedMats.push(mslBodyMat, mslConeMat, mslFlameMat);

    for (let i = 0; i < 10; i++) {
      const rootGrp = new THREE.Group();
      const defaultVisual = new THREE.Group();
      const body = new THREE.Mesh(mslBodyGeo, mslBodyMat);
      const nose = new THREE.Mesh(mslConeGeo, mslConeMat);
      nose.position.z = 0.34;
      const finH = new THREE.Mesh(mslFinGeo, mslConeMat);
      finH.position.z = -0.18;
      const finV = new THREE.Mesh(mslFinGeo, mslConeMat);
      finV.rotation.z = Math.PI / 2;
      finV.position.z = -0.18;
      defaultVisual.add(body, nose, finH, finV);
      const flame = new THREE.Mesh(mslFlameGeo, mslFlameMat);
      flame.position.z = -0.36;
      rootGrp.add(defaultVisual, flame);
      rootGrp.visible = false;
      this.scene.add(rootGrp);
      this.missiles.push({
        active: false,
        ownerIsPlayer: i < 5,
        pos: new THREE.Vector3(),
        vel: new THREE.Vector3(),
        age: 0,
        life: 0,
        damage: 40,
        smokeTimer: 0,
        isFinal: false,
        mesh: rootGrp,
        defaultVisual,
        customVisual: null,
      });
    }

    // 3D Lightning Bolt Pool for Achilles' "Удар Грозы" (20 bolts x 8 jagged segments)
    const boltSegGeo = new THREE.CylinderGeometry(0.068, 0.068, 1, 6);
    boltSegGeo.rotateX(Math.PI / 2);
    this.sharedGeos.push(boltSegGeo);
    for (let i = 0; i < 20; i++) {
      const group = new THREE.Group();
      const segments: THREE.Mesh[] = [];
      const mats: THREE.MeshBasicMaterial[] = [];
      for (let s = 0; s < 8; s++) {
        const mat = new THREE.MeshBasicMaterial({
          color: s % 3 === 0 ? '#FFFFFF' : s % 2 === 0 ? '#38BDF8' : '#FDE047',
          transparent: true,
          opacity: 0.95,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        });
        this.sharedMats.push(mat);
        const seg = new THREE.Mesh(boltSegGeo, mat);
        group.add(seg);
        segments.push(seg);
        mats.push(mat);
      }
      group.visible = false;
      this.scene.add(group);
      this.lightningBolts.push({
        active: false,
        life: 0,
        maxLife: 0.36,
        group,
        segments,
        mats,
      });
    }

    // Dynamic storm point light + ground/spear thunder aura for "Удар Грозы"
    this.thunderLight = new THREE.PointLight('#60A5FA', 0, 28, 1.35);
    this.scene.add(this.thunderLight);

    this.thunderAuraGroup = new THREE.Group();
    this.thunderAuraGroup.visible = false;

    const outerRingGeo = new THREE.RingGeometry(1.35, 1.82, 36);
    outerRingGeo.rotateX(-Math.PI / 2);
    const outerRingMat = new THREE.MeshBasicMaterial({
      color: '#38BDF8',
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(outerRingGeo);
    this.sharedMats.push(outerRingMat);
    this.thunderOuterRing = new THREE.Mesh(outerRingGeo, outerRingMat);
    this.thunderOuterRing.position.y = 0.08;

    const trimTorusGeo = new THREE.TorusGeometry(1.6, 0.036, 8, 36);
    trimTorusGeo.rotateX(Math.PI / 2);
    const trimTorusMat = new THREE.MeshBasicMaterial({
      color: '#FDE047',
      transparent: true,
      opacity: 0.82,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(trimTorusGeo);
    this.sharedMats.push(trimTorusMat);
    const trimTorus = new THREE.Mesh(trimTorusGeo, trimTorusMat);
    this.thunderOuterRing.add(trimTorus);

    const innerRingGeo = new THREE.RingGeometry(0.62, 0.96, 24);
    innerRingGeo.rotateX(-Math.PI / 2);
    const innerRingMat = new THREE.MeshBasicMaterial({
      color: '#7DD3FC',
      transparent: true,
      opacity: 0.82,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(innerRingGeo);
    this.sharedMats.push(innerRingMat);
    this.thunderInnerRing = new THREE.Mesh(innerRingGeo, innerRingMat);
    this.thunderInnerRing.position.y = 0.12;

    const pillarGeo = new THREE.CylinderGeometry(0.24, 0.68, 16, 16, 1, true);
    const pillarMat = new THREE.MeshBasicMaterial({
      color: '#38BDF8',
      transparent: true,
      opacity: 0.34,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(pillarGeo);
    this.sharedMats.push(pillarMat);
    this.thunderPillar = new THREE.Mesh(pillarGeo, pillarMat);
    this.thunderPillar.position.y = 8.0;

    const orbGeo = new THREE.SphereGeometry(0.24, 16, 16);
    const orbMat = new THREE.MeshBasicMaterial({
      color: '#E0F2FE',
      transparent: true,
      opacity: 0.96,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(orbGeo);
    this.sharedMats.push(orbMat);
    this.thunderSpearOrb = new THREE.Mesh(orbGeo, orbMat);
    this.thunderSpearOrb.visible = false;

    const glowGeo = new THREE.SphereGeometry(0.46, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
      color: '#2563EB',
      transparent: true,
      opacity: 0.62,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(glowGeo);
    this.sharedMats.push(glowMat);
    this.thunderSpearGlow = new THREE.Mesh(glowGeo, glowMat);
    this.thunderSpearOrb.add(this.thunderSpearGlow);

    const spearArcGeo = new THREE.TorusGeometry(0.44, 0.032, 8, 24);
    const spearArcMat = new THREE.MeshBasicMaterial({
      color: '#60A5FA',
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(spearArcGeo);
    this.sharedMats.push(spearArcMat);
    this.thunderSpearArcA = new THREE.Mesh(spearArcGeo, spearArcMat);
    this.thunderSpearArcB = new THREE.Mesh(spearArcGeo, spearArcMat);
    this.thunderSpearArcB.rotation.y = Math.PI / 2;
    this.thunderSpearOrb.add(this.thunderSpearArcA, this.thunderSpearArcB);

    this.scene.add(this.thunderSpearOrb);

    // Rising anime energy curtain streaks surrounding Achilles during charge (Shot 1 & 3)
    const streakGeo = new THREE.CylinderGeometry(0.045, 0.09, 6.2, 6, 1, true);
    this.sharedGeos.push(streakGeo);
    for (let i = 0; i < 12; i++) {
      const ang = (i / 12) * Math.PI * 2;
      const rad = 1.05 + (i % 3) * 0.48;
      const streakMat = new THREE.MeshBasicMaterial({
        color: i % 3 === 0 ? '#E0F2FE' : i % 2 === 0 ? '#38BDF8' : '#0EA5E9',
        transparent: true,
        opacity: 0.55,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      this.sharedMats.push(streakMat);
      const streak = new THREE.Mesh(streakGeo, streakMat);
      streak.position.set(Math.cos(ang) * rad, 1.4 + (i % 4) * 0.7, Math.sin(ang) * rad);
      streak.userData = { baseY: 1.4 + (i % 4) * 0.65, speed: 5.5 + (i % 3) * 2.1, phase: i * 0.7 };
      this.thunderAuraGroup.add(streak);
      this.thunderCurtainStreaks.push(streak);
    }

    // Blazing Lightning Lance energy thrust cone & spiral shock rings (Shot 4 & active lunge)
    const coneGeo = new THREE.ConeGeometry(0.92, 4.4, 20, 1, true);
    coneGeo.rotateX(Math.PI / 2);
    coneGeo.translate(0, 0, 2.2);
    const coneMat = new THREE.MeshBasicMaterial({
      color: '#7DD3FC',
      transparent: true,
      opacity: 0.78,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.sharedGeos.push(coneGeo);
    this.sharedMats.push(coneMat);
    this.thunderLanceCone = new THREE.Mesh(coneGeo, coneMat);
    this.thunderLanceCone.visible = false;

    for (let r = 0; r < 3; r++) {
      const ringConeGeo = new THREE.TorusGeometry(0.45 + r * 0.34, 0.042, 8, 28);
      const ringConeMat = new THREE.MeshBasicMaterial({
        color: r === 0 ? '#FFFFFF' : '#38BDF8',
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      this.sharedGeos.push(ringConeGeo);
      this.sharedMats.push(ringConeMat);
      const cRing = new THREE.Mesh(ringConeGeo, ringConeMat);
      cRing.position.z = 0.55 + r * 1.05;
      cRing.rotation.x = r * 0.22;
      cRing.rotation.y = -r * 0.18;
      this.thunderLanceCone.add(cRing);
      this.thunderConeRings.push(cRing);
    }
    this.scene.add(this.thunderLanceCone);

    this.thunderAuraGroup.add(this.thunderOuterRing, this.thunderInnerRing, this.thunderPillar);
    this.scene.add(this.thunderAuraGroup);

    const sparkGeo = new THREE.OctahedronGeometry(0.09, 0);
    const sparkMat = new THREE.MeshBasicMaterial({ color: '#FDE047' });
    this.sharedGeos.push(sparkGeo);
    this.sharedMats.push(sparkMat);

    for (let i = 0; i < 56; i++) {
      const mesh = new THREE.Mesh(sparkGeo, sparkMat);
      mesh.visible = false;
      this.scene.add(mesh);
      this.sparks.push({
        active: false,
        pos: new THREE.Vector3(),
        vel: new THREE.Vector3(),
        life: 0,
        maxLife: 0.35,
        mesh,
      });
    }

    const ringGeo = new THREE.RingGeometry(0.75, 1.0, 32);
    ringGeo.rotateX(-Math.PI / 2);
    this.sharedGeos.push(ringGeo);

    for (let i = 0; i < 8; i++) {
      const ringMat = new THREE.MeshBasicMaterial({
        color: '#F59E0B',
        transparent: true,
        opacity: 0.8,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      this.sharedMats.push(ringMat);
      const mesh = new THREE.Mesh(ringGeo, ringMat);
      mesh.visible = false;
      this.scene.add(mesh);
      this.shockwaves.push({
        active: false,
        life: 0,
        maxLife: 0.45,
        maxRadius: 3.5,
        mesh,
      });
    }
  }

  private initCameraPosition() {
    this.camOrbitYaw = Math.PI;
    this.camOrbitPitch = 0.36;
    _currentCamLook.set(0, 1.25, this.player.pos.z + 3.4);
    this.camera.position.set(0, 3.8, this.player.pos.z - 6.6);
    this.camera.lookAt(_currentCamLook);
  }

  private getCamForwardYaw(): number {
    return Math.atan2(-Math.sin(this.camOrbitYaw), -Math.cos(this.camOrbitYaw));
  }

  private getPlayerAimInfo(target: FighterRuntime): {
    camYaw: number;
    attackYaw: number;
    hasSoftLock: boolean;
  } {
    const toEnemyYaw = Math.atan2(
      target.pos.x - this.player.pos.x,
      target.pos.z - this.player.pos.z,
    );
    if (this.lockOnActive) {
      return { camYaw: toEnemyYaw, attackYaw: toEnemyYaw, hasSoftLock: true };
    }
    const camYaw = this.getCamForwardYaw();
    const diff = Math.abs(shortestAngleDiff(camYaw, toEnemyYaw));
    const hasSoftLock = diff < 0.22;
    return {
      camYaw,
      attackYaw: hasSoftLock ? toEnemyYaw : camYaw,
      hasSoftLock,
    };
  }

  private syncFreeCameraFromCurrentView() {
    const dx = this.camera.position.x - this.player.pos.x;
    const dz = this.camera.position.z - this.player.pos.z;
    const horizDist = Math.max(0.5, Math.hypot(dx, dz));
    const dy = this.camera.position.y - 1.28;
    this.camOrbitYaw = Math.atan2(dx, dz);
    this.camOrbitPitch = THREE.MathUtils.clamp(
      Math.atan2(dy, horizDist),
      -0.1,
      0.92,
    );
  }

  public setMouseSensitivity(val: number) {
    this.mouseSensitivity = THREE.MathUtils.clamp(val, 0.2, 2.5);
  }

  public requestPointerLock() {
    if (this.disposed || this.paused || this.outcome !== 'ongoing') return;
    const dom = this.renderer.domElement;
    if (document.pointerLockElement === dom) return;
    try {
      const maybePromise = dom.requestPointerLock?.() as unknown;
      if (maybePromise && typeof (maybePromise as Promise<void>).catch === 'function') {
        (maybePromise as Promise<void>).catch(() => {});
      }
    } catch {
      // Pointer lock may be blocked outside user gesture or in strict sandbox
    }
  }

  public exitPointerLock() {
    const dom = this.renderer.domElement;
    if (document.pointerLockElement === dom) {
      try {
        document.exitPointerLock?.();
      } catch {
        // Ignore
      }
    }
  }

  private updateCursorVisibility() {
    const hideCursor = !this.paused && this.outcome === 'ongoing';
    this.renderer.domElement.style.cursor = hideCursor ? 'none' : 'default';
  }

  private onPointerLockChange = () => {
    const dom = this.renderer.domElement;
    const locked = document.pointerLockElement === dom;
    this.callbacks.onPointerLockChange?.(locked);
    if (this.wasPointerLocked && !locked && !this.paused && this.outcome === 'ongoing') {
      this.callbacks.onRequestPause?.();
    }
    this.wasPointerLocked = locked;
  };

  private onKeyDown = (e: KeyboardEvent) => {
    const code = e.code;
    this.keys.add(code);
    if (code === 'KeyJ') this.triggerPlayerAttack();
    if (code === 'KeyE' || code === 'KeyF') this.triggerPlayerSpecial();
    if (code === 'KeyP') {
      this.callbacks.onRequestPause?.();
    }
    if (code === 'Space') {
      e.preventDefault();
      this.triggerPlayerJump();
    }
    if (code === 'ShiftLeft' || code === 'ShiftRight') {
      e.preventDefault();
      this.triggerPlayerDash();
    }
    if (code === 'Tab' || code === 'KeyR') {
      e.preventDefault();
      this.toggleLockOn();
    }
  };

  private onKeyUp = (e: KeyboardEvent) => {
    this.keys.delete(e.code);
  };

  private onMouseDown = (e: MouseEvent) => {
    if (!this.paused && this.outcome === 'ongoing') {
      this.requestPointerLock();
    }
    if (e.button === 0) {
      this.mouseAttackQueued = true;
    } else if (e.button === 2) {
      this.mouseGuardHeld = true;
    }
  };

  private onMouseUp = (e: MouseEvent) => {
    if (e.button === 2) {
      this.mouseGuardHeld = false;
    }
  };

  private onMouseEnter = (e: MouseEvent) => {
    this.lastPointerX = e.clientX;
    this.lastPointerY = e.clientY;
    this.pointerEdgeRate = 0;
  };

  private onMouseLeave = () => {
    this.lastPointerX = null;
    this.lastPointerY = null;
    this.pointerEdgeRate = 0;
  };

  private onMouseMove = (e: MouseEvent) => {
    if (this.lockOnActive || this.paused) {
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
      this.pointerEdgeRate = 0;
      return;
    }

    const dom = this.renderer.domElement;
    let dx = 0;
    let dy = 0;
    if (document.pointerLockElement === dom) {
      dx = THREE.MathUtils.clamp(e.movementX, -180, 180);
      dy = THREE.MathUtils.clamp(e.movementY, -180, 180);
      this.pointerEdgeRate = 0;
    } else {
      if (this.lastPointerX !== null && this.lastPointerY !== null) {
        dx = THREE.MathUtils.clamp(e.clientX - this.lastPointerX, -48, 48);
        dy = THREE.MathUtils.clamp(e.clientY - this.lastPointerY, -48, 48);
      }
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;

      // Gentle continuous turn when cursor reaches left/right outer margin
      const rect = dom.getBoundingClientRect();
      const edgeZone = 42;
      if (e.clientX <= rect.left + edgeZone && dx <= 0) {
        const t = 1 - Math.max(0, e.clientX - rect.left) / edgeZone;
        this.pointerEdgeRate = 1.65 * t;
      } else if (e.clientX >= rect.right - edgeZone && dx >= 0) {
        const t = 1 - Math.max(0, rect.right - e.clientX) / edgeZone;
        this.pointerEdgeRate = -1.65 * t;
      } else {
        this.pointerEdgeRate = 0;
      }
    }

    const sens = this.mouseSensitivity * (this.player.isAiming ? 0.46 : 1.0);
    this.camOrbitYaw -= dx * 0.0052 * sens;
    this.camOrbitPitch = THREE.MathUtils.clamp(
      this.camOrbitPitch + dy * 0.0038 * sens,
      -0.1,
      0.92,
    );
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.camDistance = THREE.MathUtils.clamp(
      this.camDistance + e.deltaY * 0.0038,
      4.6,
      10.6,
    );
  };

  private onTouchStart = (e: TouchEvent) => {
    if (this.activeTouchId === null && e.changedTouches.length > 0) {
      const t = e.changedTouches[0];
      this.activeTouchId = t.identifier;
      this.lastTouchX = t.clientX;
      this.lastTouchY = t.clientY;
    }
  };

  private onTouchMove = (e: TouchEvent) => {
    if (this.lockOnActive || this.paused || this.activeTouchId === null) return;
    for (let i = 0; i < e.changedTouches.length; i++) {
      const t = e.changedTouches[i];
      if (t.identifier === this.activeTouchId) {
        const dx = THREE.MathUtils.clamp(t.clientX - this.lastTouchX, -48, 48);
        const dy = THREE.MathUtils.clamp(t.clientY - this.lastTouchY, -48, 48);
        this.lastTouchX = t.clientX;
        this.lastTouchY = t.clientY;
        this.camOrbitYaw -= dx * 0.007;
        this.camOrbitPitch = THREE.MathUtils.clamp(
          this.camOrbitPitch + dy * 0.005,
          -0.1,
          0.92,
        );
        break;
      }
    }
  };

  private onTouchEnd = (e: TouchEvent) => {
    for (let i = 0; i < e.changedTouches.length; i++) {
      if (e.changedTouches[i].identifier === this.activeTouchId) {
        this.activeTouchId = null;
        break;
      }
    }
  };

  private onContextMenu = (e: MouseEvent) => {
    e.preventDefault();
  };

  private onResize = () => {
    if (this.disposed) return;
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  private bindEvents() {
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('resize', this.onResize);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);
    const dom = this.renderer.domElement;
    dom.addEventListener('mousedown', this.onMouseDown);
    window.addEventListener('mouseup', this.onMouseUp);
    dom.addEventListener('mouseenter', this.onMouseEnter);
    dom.addEventListener('mouseleave', this.onMouseLeave);
    dom.addEventListener('mousemove', this.onMouseMove);
    dom.addEventListener('wheel', this.onWheel, { passive: false });
    dom.addEventListener('touchstart', this.onTouchStart, { passive: true });
    dom.addEventListener('touchmove', this.onTouchMove, { passive: true });
    dom.addEventListener('touchend', this.onTouchEnd);
    dom.addEventListener('contextmenu', this.onContextMenu);
  }

  public setPaused(val: boolean) {
    this.paused = val;
    this.updateCursorVisibility();
    if (val) {
      this.exitPointerLock();
    } else {
      this.clock.getDelta();
    }
  }

  public setVirtualMove(x: number, z: number) {
    this.virtualMove.x = x;
    this.virtualMove.z = z;
  }

  public triggerPlayerAttack() {
    this.mouseAttackQueued = true;
  }

  public triggerPlayerGuard(held: boolean) {
    this.mouseGuardHeld = held;
  }

  public triggerPlayerJump() {
    if (this.outcome !== 'ongoing' || this.paused) return;
    this.tryJump(this.player);
  }

  public triggerPlayerDash() {
    if (this.outcome !== 'ongoing' || this.paused) return;
    this.tryDash(this.player, this.getPlayerMoveInput());
  }

  public triggerPlayerSpecial() {
    if (this.outcome !== 'ongoing' || this.paused) return;
    this.trySpecial(this.player, this.enemy);
  }

  public toggleLockOn() {
    this.lockOnActive = !this.lockOnActive;
    if (!this.lockOnActive) {
      this.syncFreeCameraFromCurrentView();
    }
    soundEngine.playClick();
  }

  private getPlayerMoveInput(): { x: number; z: number } {
    let ix = this.virtualMove.x;
    let iz = this.virtualMove.z;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) iz -= 1;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) iz += 1;
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) ix -= 1;
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) ix += 1;

    const len = Math.hypot(ix, iz);
    if (len < 0.01) return { x: 0, z: 0 };
    ix /= len;
    iz /= len;

    if (!this.lockOnActive) {
      _camForward.set(-Math.sin(this.camOrbitYaw), 0, -Math.cos(this.camOrbitYaw));
    } else {
      this.camera.getWorldDirection(_camForward);
      _camForward.y = 0;
      _camForward.normalize();
    }
    _camRight.set(-_camForward.z, 0, _camForward.x);

    const worldX = _camForward.x * -iz + _camRight.x * ix;
    const worldZ = _camForward.z * -iz + _camRight.z * ix;
    return { x: worldX, z: worldZ };
  }

  private tryJump(f: FighterRuntime) {
    if (f.stunTimer > 0 || !f.isGrounded || f.hp <= 0) return;
    f.isGrounded = false;
    f.velY = f.stats.jumpPower;
    f.isGuarding = false;
    soundEngine.playJump();
    this.spawnSparks(
      f.pos.x,
      f.pos.y + 0.14,
      f.pos.z,
      f.stats.jumpPower > 10.2 ? 7 : 5,
      f.weapon.accentHex,
    );
  }

  private tryDash(f: FighterRuntime, moveDir: { x: number; z: number }) {
    if (f.stunTimer > 0 || f.isDashing || f.energy < 18) return;
    f.energy -= 18;
    f.isDashing = true;
    f.dashTimer = f.stats.dashDuration;
    f.invulnTimer = 0.16;
    f.isGuarding = false;
    f.attackPhase = 'none';
    f.airAttacking = false;

    if (Math.hypot(moveDir.x, moveDir.z) > 0.1) {
      f.dashDir.set(moveDir.x, 0, moveDir.z).normalize();
    } else {
      f.dashDir.set(Math.sin(f.yaw), 0, Math.cos(f.yaw));
    }
    soundEngine.playDash();
    this.spawnSparks(f.pos.x, f.pos.y + 0.7, f.pos.z, 6, f.weapon.accentHex);
  }

  private tryAttack(f: FighterRuntime, target: FighterRuntime) {
    if (f.stunTimer > 0 || f.isDashing || f.attackCooldown > 0 || f.attackPhase !== 'none') {
      return;
    }

    if (f.isPlayer && !this.lockOnActive) {
      f.targetYaw = this.getPlayerAimInfo(target).attackYaw;
    } else {
      const dx = target.pos.x - f.pos.x;
      const dz = target.pos.z - f.pos.z;
      f.targetYaw = Math.atan2(dx, dz);
    }

    f.isGuarding = false;
    f.comboStep = (f.comboStep % 3) + 1;
    f.comboResetTimer = 1.15;
    f.hasHitCurrentSwing = false;
    f.isSpecialAttack = false;
    f.airAttacking = !f.isGrounded;

    if (f.airAttacking && !f.weapon.isRanged && f.velY < 2.2) {
      // Brief mid-air coil/hang before plunging strike
      f.velY = 2.2;
    }

    const tempo = f.weapon.attackSpeed;
    const baseDur = f.weapon.isRanged
      ? 0.34
      : f.airAttacking
        ? 0.46
        : f.weapon.kind === 'lance'
          ? 0.52
          : 0.44;
    f.attackDuration = baseDur / tempo;
    f.attackTimer = f.attackDuration;
    f.attackPhase = 'windup';
    f.attackCooldown = f.attackDuration * 0.92;

    if (f.weapon.isRanged) {
      soundEngine.playLaser(false);
    } else {
      soundEngine.playSlash(f.comboStep === 3 || f.airAttacking);
    }
  }

  private trySpecial(f: FighterRuntime, target: FighterRuntime) {
    if (f.special < 99.9 || f.stunTimer > 0) return;
    f.special = 0;
    f.isGuarding = false;
    f.isAiming = false;
    f.isDashing = false;
    f.hasHitCurrentSwing = false;
    f.isSpecialAttack = true;
    f.specialStage = 0;
    f.airAttacking = !f.isGrounded;
    f.attackPhase = 'windup';

    const dur =
      f.weapon.kind === 'kodachi'
        ? 1.18
        : f.weapon.kind === 'hammer'
          ? 1.04
          : f.weapon.kind === 'lance'
            ? 4.95
            : f.weapon.isRanged
              ? 0.92
              : 0.72;
    f.attackDuration = dur;
    f.attackTimer = dur;
    f.attackCooldown = dur + 0.08;
    f.invulnTimer = f.weapon.kind === 'lance' ? 4.5 : 0.48;

    if (f.weapon.kind === 'hammer' && f.isGrounded) {
      f.isGrounded = false;
      f.velY = 6.2;
      f.pos.y += 0.08;
    }

    if (f.isPlayer && !this.lockOnActive) {
      f.targetYaw = this.getPlayerAimInfo(target).attackYaw;
    } else {
      const dx = target.pos.x - f.pos.x;
      const dz = target.pos.z - f.pos.z;
      f.targetYaw = Math.atan2(dx, dz);
    }

    if (f.weapon.kind === 'lance') {
      startCustomSpecialClip(f.rig, dur * 0.68);
      this.thunderAuraTimer = dur * 0.96;
      this.thunderFlashIntensity = 15;
      this.lastThunderShot = -1;
      this.thunderShotActive = false;
      this.thunderShotTimer = 0;
      this.thunderShotCaster = null;
      this.thunderShotTarget = null;
      this.thunderLight.position.set(f.pos.x, f.pos.y + 1.8, f.pos.z);
      soundEngine.playThunderCharge();
    }

    if (f.isPlayer) {
      this.specialsUsed += 1;
    }
    this.bannerText = `${f.name.split(' ')[0]}: ${f.weapon.specialName.toUpperCase()}!`;
    this.bannerTimer = f.weapon.kind === 'lance' ? 4.95 : 1.75;
    soundEngine.playSpecial();
    this.spawnShockwave(f.pos.x, f.pos.y + 0.15, f.pos.z, 2.9, f.weapon.accentHex);
  }

  private damageDestructibleObstacle(obsState: ObstacleState, damage: number, attacker: FighterRuntime) {
    if (obsState.destroyed || !obsState.data.destructible) return;
    obsState.hp -= damage;
    this.spawnSparks(obsState.data.x, 0.85, obsState.data.z, 6, '#F59E0B');

    if (obsState.hp <= 0) {
      obsState.destroyed = true;
      const vis = this.obstacleVisuals.get(obsState.id);
      if (vis) {
        vis.destroyed = true;
        vis.group.visible = false;
      }
      soundEngine.playHit(true);
      this.spawnSparks(obsState.data.x, 0.9, obsState.data.z, 16, '#F59E0B');
      this.spawnShockwave(obsState.data.x, 0.12, obsState.data.z, 2.2, '#F59E0B');

      // Supply crate drops immediate repair/energy pickup for breaker!
      attacker.hp = Math.min(attacker.stats.maxHp, attacker.hp + 55);
      attacker.energy = Math.min(attacker.stats.maxEnergy, attacker.energy + 28);
      attacker.special = Math.min(100, attacker.special + 14);
      if (attacker.isPlayer) {
        this.bannerText = 'ЯЩИК РАЗРУШЕН: +55 HP и +28 Энергии!';
        this.bannerTimer = 1.4;
      }
    }
  }

  private spawnProjectile(
    owner: FighterRuntime,
    target: FighterRuntime,
    damage: number,
    speed: number,
    isSpecial: boolean,
    spreadYaw = 0,
  ) {
    const slot = this.projectiles.find(
      (p) => !p.active && p.ownerIsPlayer === owner.isPlayer,
    );
    if (!slot) return;

    owner.rig.muzzlePoint.getWorldPosition(_v1);
    if (_v1.y < owner.pos.y + 0.4) {
      _v1.set(owner.pos.x, owner.pos.y + 1.15, owner.pos.z);
    }

    if (owner.isPlayer && !this.lockOnActive) {
      this.camera.getWorldDirection(_v2);
      const aimPoint = _v3.copy(this.camera.position).addScaledVector(_v2, 65);
      aimPoint.y = Math.max(0.25, aimPoint.y);

      const targetCenter = new THREE.Vector3(target.pos.x, target.pos.y + 1.02, target.pos.z);
      const rayToTarget = targetCenter.clone().sub(this.camera.position);
      const projAlongRay = rayToTarget.dot(_v2);
      const aimInfo = this.getPlayerAimInfo(target);
      const perpDist =
        projAlongRay > 0
          ? rayToTarget.addScaledVector(_v2, -projAlongRay).length()
          : Infinity;
      if (aimInfo.hasSoftLock || (owner.isAiming && perpDist < 1.15)) {
        aimPoint.copy(targetCenter);
      }

      const dir = aimPoint.sub(_v1).normalize();
      if (spreadYaw !== 0) {
        dir.applyAxisAngle(new THREE.Vector3(0, 1, 0), spreadYaw);
      }
      slot.vel.copy(dir).multiplyScalar(speed);
    } else {
      const dx = target.pos.x - _v1.x;
      const dy = target.pos.y + 1.05 - _v1.y;
      const dz = target.pos.z - _v1.z;
      const dir = _v3.set(dx, dy, dz);
      if (dir.lengthSq() > 0.0001) {
        dir.normalize();
      } else {
        dir.set(0, 0, 1);
      }
      if (spreadYaw !== 0) {
        dir.applyAxisAngle(new THREE.Vector3(0, 1, 0), spreadYaw);
      }
      slot.vel.copy(dir).multiplyScalar(speed);
    }

    slot.active = true;
    slot.pos.copy(_v1);
    slot.prevPos.copy(_v1);
    slot.damage = damage;
    slot.life = 2.2;
    slot.isSpecial = isSpecial;

    slot.mesh.position.copy(slot.pos);
    slot.mesh.quaternion.setFromUnitVectors(_zAxis, _v2.copy(slot.vel).normalize());
    slot.mesh.scale.setScalar(isSpecial ? 1.5 : owner.isAiming ? 1.18 : 1.0);
    slot.mesh.visible = true;

    // Bright muzzle flash sparks at barrel tip
    this.spawnSparks(
      _v1.x,
      _v1.y,
      _v1.z,
      isSpecial ? 10 : owner.isAiming ? 7 : 5,
      isSpecial ? '#FDE047' : '#38BDF8',
    );
  }

  private spawnLightningBolt(
    fromX: number,
    fromY: number,
    fromZ: number,
    toX: number,
    toY: number,
    toZ: number,
    thickness = 1.0,
    colorHex = '#7DD3FC',
  ) {
    const slot = this.lightningBolts.find((b) => !b.active);
    if (!slot) return;
    slot.active = true;
    slot.maxLife = 0.36;
    slot.life = slot.maxLife;
    slot.group.visible = true;

    const start = new THREE.Vector3(fromX, fromY, fromZ);
    const end = new THREE.Vector3(toX, toY, toZ);
    const dir = new THREE.Vector3().subVectors(end, start);
    const totalLen = Math.max(0.2, dir.length());
    const count = slot.segments.length;
    const pts: THREE.Vector3[] = [start];

    for (let i = 1; i < count; i += 1) {
      const t = i / count;
      const p = new THREE.Vector3().lerpVectors(start, end, t);
      const jitter = Math.min(0.65, totalLen * 0.1) * (1 - Math.abs(t - 0.5) * 0.8);
      p.x += (Math.random() - 0.5) * 2 * jitter;
      p.y += (Math.random() - 0.5) * 1.4 * jitter;
      p.z += (Math.random() - 0.5) * 2 * jitter;
      pts.push(p);
    }
    pts.push(end);

    for (let i = 0; i < count; i += 1) {
      const seg = slot.segments[i];
      const mat = slot.mats[i];
      const a = pts[i];
      const b = pts[i + 1];
      const segVec = _v2.subVectors(b, a);
      const len = Math.max(0.05, segVec.length());
      seg.position.copy(a).add(b).multiplyScalar(0.5);
      seg.quaternion.setFromUnitVectors(_zAxis, segVec.normalize());
      seg.scale.set(thickness, thickness, len);
      mat.color.set(i % 2 === 0 ? colorHex : '#FDE047');
      mat.opacity = 0.96;
    }
  }

  private launchStingerMissile(
    owner: FighterRuntime,
    target: FighterRuntime,
    slotIdx: number,
    isFinal: boolean,
  ) {
    const slot = this.missiles.find((m) => !m.active);
    if (!slot) return;

    if (slot.customVisual) {
      slot.mesh.remove(slot.customVisual);
      slot.customVisual = null;
    }

    const fx = Math.sin(owner.yaw);
    const fz = Math.cos(owner.yaw);
    const rx = -fz;
    const rz = fx;
    const sideSign = slotIdx % 2 === 0 ? -1 : 1;

    const backList = owner.rig.backMissiles ?? [];
    if (backList.length > 0) {
      const src = backList[slotIdx % backList.length];
      src.getWorldPosition(_v1);
      const wScale = new THREE.Vector3();
      src.getWorldScale(wScale);
      src.visible = false;
      owner.missileRearmTimer = 1.65;

      const cloned = src.clone();
      cloned.visible = true;
      cloned.position.set(0, 0, 0);
      cloned.rotation.set(0, 0, 0);
      cloned.scale.copy(wScale);
      slot.defaultVisual.visible = false;
      slot.mesh.add(cloned);
      slot.customVisual = cloned;
    } else {
      slot.defaultVisual.visible = true;
      _v1.set(
        owner.pos.x - fx * 0.3 + rx * sideSign * (0.28 + (slotIdx % 3) * 0.14),
        owner.pos.y + 1.45 + Math.floor(slotIdx / 2) * 0.14,
        owner.pos.z - fz * 0.3 + rz * sideSign * (0.28 + (slotIdx % 3) * 0.14),
      );
    }

    slot.active = true;
    slot.ownerIsPlayer = owner.isPlayer;
    slot.pos.copy(_v1);
    slot.vel.set(
      fx * 3.8 + rx * sideSign * (5.2 + (slotIdx % 3) * 1.4),
      8.2 + (slotIdx % 2) * 2.0,
      fz * 3.8 + rz * sideSign * (5.2 + (slotIdx % 3) * 1.4),
    );
    slot.age = 0;
    slot.life = 2.6;
    slot.damage = Math.round(owner.weapon.damage * 0.56);
    slot.smokeTimer = 0;
    slot.isFinal = isFinal;

    slot.mesh.position.copy(slot.pos);
    slot.mesh.quaternion.setFromUnitVectors(_zAxis, _v2.copy(slot.vel).normalize());
    slot.mesh.visible = true;

    soundEngine.playLaser(isFinal);
    this.spawnSparks(_v1.x, _v1.y, _v1.z, 6, '#FB923C');
    void target;
  }

  private spawnSparks(
    x: number,
    y: number,
    z: number,
    count: number,
    colorHex: string,
  ) {
    let spawned = 0;
    for (let i = 0; i < this.sparks.length && spawned < count; i++) {
      const s = this.sparks[i];
      if (!s.active) {
        s.active = true;
        s.pos.set(x, y, z);
        const angle = Math.random() * Math.PI * 2;
        const spd = 2.5 + Math.random() * 5.5;
        s.vel.set(
          Math.cos(angle) * spd,
          2.0 + Math.random() * 4.5,
          Math.sin(angle) * spd,
        );
        s.maxLife = 0.25 + Math.random() * 0.2;
        s.life = s.maxLife;
        (s.mesh.material as THREE.MeshBasicMaterial).color.set(colorHex);
        s.mesh.position.copy(s.pos);
        s.mesh.visible = true;
        spawned++;
      }
    }
  }

  private spawnShockwave(
    x: number,
    y: number,
    z: number,
    maxRadius: number,
    colorHex: string,
  ) {
    const ring = this.shockwaves.find((r) => !r.active);
    if (!ring) return;
    ring.active = true;
    ring.maxLife = 0.42;
    ring.life = 0.42;
    ring.maxRadius = maxRadius;
    ring.mesh.position.set(x, y, z);
    ring.mesh.scale.setScalar(0.2);
    (ring.mesh.material as THREE.MeshBasicMaterial).color.set(colorHex);
    ring.mesh.visible = true;
  }

  private applyHit(
    attacker: FighterRuntime,
    defender: FighterRuntime,
    rawDamage: number,
    knockback: number,
    isHeavy: boolean,
  ) {
    if (defender.invulnTimer > 0 || this.outcome !== 'ongoing') return;

    const toAttackerX = attacker.pos.x - defender.pos.x;
    const toAttackerZ = attacker.pos.z - defender.pos.z;
    const len = Math.max(0.01, Math.hypot(toAttackerX, toAttackerZ));
    const facingDot =
      Math.sin(defender.yaw) * (toAttackerX / len) +
      Math.cos(defender.yaw) * (toAttackerZ / len);

    const guardEffective = defender.isGuarding && facingDot > -0.15;

    if (guardEffective) {
      const heldFor = this.elapsedTime - defender.guardStartTime;
      const isParry = heldFor <= 0.26 && !isHeavy;

      if (isParry) {
        soundEngine.playBlock(true);
        defender.special = Math.min(100, defender.special + 24);
        attacker.stunTimer = 0.58;
        attacker.attackPhase = 'none';
        if (defender.isPlayer) {
          this.parriesLanded += 1;
          this.bannerText = 'ИДЕАЛЬНОЕ ПАРИРОВАНИЕ!';
          this.bannerTimer = 1.1;
        }
        this.spawnSparks(defender.pos.x, 1.1, defender.pos.z, 12, '#38BDF8');
        this.spawnShockwave(defender.pos.x, 0.9, defender.pos.z, 2.0, '#38BDF8');
        return;
      } else {
        const mitigated = rawDamage * (1 - defender.weapon.blockMitigation);
        const dmg = Math.max(4, Math.round(mitigated));
        defender.hp = Math.max(0, defender.hp - dmg);
        defender.energy = Math.max(0, defender.energy - 8);
        defender.vel.x -= (toAttackerX / len) * knockback * 0.45;
        defender.vel.z -= (toAttackerZ / len) * knockback * 0.45;
        soundEngine.playBlock(false);
        this.spawnSparks(defender.pos.x, 1.05, defender.pos.z, 7, '#94A3B8');
        if (attacker.isPlayer) this.damageDealt += dmg;
        this.checkKo();
        return;
      }
    }

    const powerScale = attacker.stats.attackPower / 100;
    const armorFactor = 1 - defender.stats.armor / 100;
    const finalDmg = Math.max(
      10,
      Math.round(rawDamage * powerScale * armorFactor),
    );

    defender.hp = Math.max(0, defender.hp - finalDmg);
    defender.hitFlashTimer = 0.14;
    defender.stunTimer = isHeavy ? 0.34 : 0.18;
    if (isHeavy) {
      defender.attackPhase = 'none';
    }
    defender.vel.x -= (toAttackerX / len) * knockback;
    defender.vel.z -= (toAttackerZ / len) * knockback;

    attacker.special = Math.min(100, attacker.special + (isHeavy ? 18 : 12));
    defender.special = Math.min(100, defender.special + 9);
    attacker.comboCounter += 1;
    attacker.comboDecayTimer = 2.4;

    if (attacker.isPlayer) {
      this.damageDealt += finalDmg;
    }

    this.cameraShake = Math.min(0.45, this.cameraShake + (isHeavy ? 0.25 : 0.11));
    soundEngine.playHit(isHeavy);
    this.spawnSparks(
      defender.pos.x,
      1.05,
      defender.pos.z,
      isHeavy ? 14 : 8,
      attacker.weapon.accentHex,
    );
    if (isHeavy) {
      this.spawnShockwave(
        defender.pos.x,
        0.18,
        defender.pos.z,
        2.6,
        attacker.weapon.accentHex,
      );
    }

    this.checkKo();
  }

  private checkKo() {
    if (this.outcome !== 'ongoing') return;
    if (this.enemy.hp <= 0) {
      this.outcome = 'victory';
      this.outcomeDelayTimer = 1.6;
      this.bannerText = 'ПОБЕДА! LBX ПРОТИВНИКА ПОВЕРЖЕН';
      this.bannerTimer = 3.0;
      this.updateCursorVisibility();
      this.exitPointerLock();
      this.spawnShockwave(this.enemy.pos.x, 0.3, this.enemy.pos.z, 4.5, '#F59E0B');
      soundEngine.playOutcome(true);
    } else if (this.player.hp <= 0) {
      this.outcome = 'defeat';
      this.outcomeDelayTimer = 1.6;
      this.bannerText = 'ПОРАЖЕНИЕ — LBX ВЫВЕДЕН ИЗ СТРОЯ';
      this.bannerTimer = 3.0;
      this.updateCursorVisibility();
      this.exitPointerLock();
      this.spawnShockwave(this.player.pos.x, 0.3, this.player.pos.z, 4.5, '#EF4444');
      soundEngine.playOutcome(false);
    }
  }

  private updateInteractables(dt: number) {
    this.readyInteractablesScratch.length = 0;

    for (let i = 0; i < this.interactableStates.length; i++) {
      const item = this.interactableStates[i];
      if (item.cooldown > 0) {
        item.cooldown = Math.max(0, item.cooldown - dt);
      }

      const vis = this.interactableVisuals.get(item.data.id);
      if (vis) {
        const ready = item.cooldown <= 0;
        vis.beaconMesh.visible = ready;
        vis.beaconMesh.rotation.y += dt * 1.8;
        vis.beaconMesh.position.y = 0.72 + Math.sin(this.elapsedTime * 3.0 + i) * 0.12;
        (vis.ringMesh.material as THREE.MeshBasicMaterial).opacity = ready ? 0.92 : 0.18;
      }

      if (item.cooldown <= 0 && this.outcome === 'ongoing') {
        this.readyInteractablesScratch.push(item.data);

        const fighters = [this.player, this.enemy];
        for (let fIdx = 0; fIdx < fighters.length; fIdx++) {
          const f = fighters[fIdx];
          if (f.hp <= 0) continue;
          const dist = Math.hypot(f.pos.x - item.data.x, f.pos.z - item.data.z);
          if (dist <= item.data.radius + 0.55) {
            if (item.data.kind === 'repair_node') {
              f.hp = Math.min(f.stats.maxHp, f.hp + 150);
              f.special = Math.min(100, f.special + 22);
              item.cooldown = 11.0;
              soundEngine.playSpecial();
              this.spawnShockwave(item.data.x, 0.15, item.data.z, 2.4, '#34D399');
              if (f.isPlayer) {
                this.bannerText = 'РЕМОНТНЫЙ УЗЕЛ: +150 HP и +22% Экстра!';
                this.bannerTimer = 1.4;
              }
            } else {
              f.energy = Math.min(f.stats.maxEnergy, f.energy + 45);
              // Smooth boost surge forward
              f.vel.x += Math.sin(f.yaw) * 7.5;
              f.vel.z += Math.cos(f.yaw) * 7.5;
              item.cooldown = 8.0;
              soundEngine.playDash();
              this.spawnShockwave(item.data.x, 0.15, item.data.z, 2.2, '#38BDF8');
              if (f.isPlayer) {
                this.bannerText = 'ЭНЕРГО-УСКОРИТЕЛЬ: +45 Буст-энергии!';
                this.bannerTimer = 1.3;
              }
            }
            break;
          }
        }
      }
    }
  }

  private updateFighter(
    f: FighterRuntime,
    opponent: FighterRuntime,
    moveInput: { x: number; z: number },
    wantAttack: boolean,
    wantGuard: boolean,
    wantDash: boolean,
    wantSpecial: boolean,
    aimAngle: number | null,
    dt: number,
  ) {
    f.attackCooldown = Math.max(0, f.attackCooldown - dt);
    f.stunTimer = Math.max(0, f.stunTimer - dt);
    f.hitFlashTimer = Math.max(0, f.hitFlashTimer - dt);
    f.invulnTimer = Math.max(0, f.invulnTimer - dt);
    f.jumpSquashTimer = Math.max(0, f.jumpSquashTimer - dt);
    f.comboResetTimer = Math.max(0, f.comboResetTimer - dt);
    if (f.missileRearmTimer > 0) {
      f.missileRearmTimer = Math.max(0, f.missileRearmTimer - dt);
      if (f.missileRearmTimer <= 0 && f.rig.backMissiles) {
        f.rig.backMissiles.forEach((m) => {
          m.visible = true;
        });
      }
    }
    if (f.comboResetTimer <= 0 && f.attackPhase === 'none') {
      f.comboStep = 0;
    }
    if (f.attackPhase === 'none') {
      f.airAttacking = false;
    }
    f.comboDecayTimer = Math.max(0, f.comboDecayTimer - dt);
    if (f.comboDecayTimer <= 0) {
      f.comboCounter = 0;
    }

    f.rig.hitFlashMaterials.forEach((mat) => {
      mat.emissive.setHex(f.hitFlashTimer > 0 ? 0xff4422 : 0x000000);
      mat.emissiveIntensity = f.hitFlashTimer > 0 ? 0.65 : 0;
    });

    if (f.hp <= 0) {
      f.isAiming = false;
      stopCustomSpecialClip(f.rig);
      f.vel.x = THREE.MathUtils.damp(f.vel.x, 0, 8, dt);
      f.vel.z = THREE.MathUtils.damp(f.vel.z, 0, 8, dt);
      if (f.pos.y > 0) {
        f.velY -= 24.5 * dt;
        f.pos.y = Math.max(0, f.pos.y + f.velY * dt);
      }
      f.pos.x += f.vel.x * dt;
      f.pos.z += f.vel.z * dt;
      f.rig.root.position.copy(f.pos);
      f.rig.shadowDisc.position.y = 0.015 - f.pos.y;
      f.rig.bodyGroup.rotation.x = THREE.MathUtils.damp(
        f.rig.bodyGroup.rotation.x,
        -Math.PI * 0.45,
        6,
        dt,
      );
      f.rig.bodyGroup.position.y = THREE.MathUtils.damp(
        f.rig.bodyGroup.position.y,
        -0.45,
        6,
        dt,
      );
      return;
    }

    if (f.stunTimer > 0 && f.rig.specialAnimActive) {
      stopCustomSpecialClip(f.rig);
    }

    if (!f.isDashing && !f.isGuarding) {
      f.energy = Math.min(
        f.stats.maxEnergy,
        f.energy + f.stats.energyRegen * dt,
      );
    }

    if (wantSpecial) {
      this.trySpecial(f, opponent);
    } else if (wantDash) {
      this.tryDash(f, moveInput);
    } else if (wantAttack) {
      this.tryAttack(f, opponent);
    }

    const canGuard =
      f.stats.canGuard &&
      wantGuard &&
      f.isGrounded &&
      !f.isDashing &&
      f.attackPhase === 'none' &&
      f.stunTimer <= 0 &&
      f.energy > 4;
    if (canGuard && !f.isGuarding) {
      f.guardStartTime = this.elapsedTime;
    }
    f.isGuarding = canGuard;
    if (f.isGuarding) {
      f.energy = Math.max(0, f.energy - 6.5 * dt);
    }

    f.isAiming =
      !f.stats.canGuard &&
      f.weapon.isRanged &&
      wantGuard &&
      !f.isDashing &&
      !f.isSpecialAttack &&
      f.stunTimer <= 0 &&
      f.hp > 0;

    const usesPhysicalShield =
      f.weapon.kind === 'lance' ||
      Boolean(f.rig.isCustomModel && !f.rig.hasHangarWeapon && f.rig.hasBuiltInWeapon);
    f.rig.guardBubble.visible = f.isGuarding && !usesPhysicalShield;
    if (f.rig.guardBubble.visible) {
      f.rig.guardBubble.rotation.y += dt * 1.8;
    }

    if (f.attackPhase !== 'none') {
      f.attackTimer -= dt;
      const progress = 1 - Math.max(0, f.attackTimer / f.attackDuration);

      if (f.isSpecialAttack) {
        if (f.weapon.isRanged) {
          // Hunter "Залп Стингера": launch 6 back-mounted Stinger homing missiles in rapid sequence
          const salvoTimes = [0.14, 0.23, 0.32, 0.41, 0.50, 0.59];
          while (f.specialStage < salvoTimes.length && progress >= salvoTimes[f.specialStage]) {
            this.launchStingerMissile(
              f,
              opponent,
              f.specialStage,
              f.specialStage === salvoTimes.length - 1,
            );
            f.specialStage += 1;
          }
          f.attackPhase = progress < 0.14 ? 'windup' : progress < 0.72 ? 'active' : 'recover';
        } else if (f.weapon.kind === 'kodachi') {
          // Kunoichi "Быстрый как ветер": 3 rapid horizontal cyclone strikes + fierce bottom-to-top upward launch
          const windTimes = [0.16, 0.34, 0.52, 0.78];
          while (f.specialStage < windTimes.length && progress >= windTimes[f.specialStage]) {
            const stage = f.specialStage;
            f.specialStage += 1;
            const fx = Math.sin(f.yaw);
            const fz = Math.cos(f.yaw);
            const isUppercut = stage === 3;
            f.vel.x += fx * (isUppercut ? 5.8 : 9.2);
            f.vel.z += fz * (isUppercut ? 5.8 : 9.2);
            soundEngine.playSlash(isUppercut || stage === 2);

            const dx = opponent.pos.x - f.pos.x;
            const dz = opponent.pos.z - f.pos.z;
            const dist = Math.hypot(dx, dz);
            const dot = dist > 0.01 ? (fx * dx + fz * dz) / dist : 1;
            if (dist <= f.weapon.range + 1.55 && dot > -0.15 && Math.abs(opponent.pos.y - f.pos.y) <= 2.4) {
              if (!isUppercut) {
                this.applyHit(f, opponent, f.weapon.damage * 0.46, 1.5, false);
                if (opponent.hp > 0) {
                  opponent.stunTimer = Math.max(opponent.stunTimer, 0.44);
                }
              } else {
                this.applyHit(f, opponent, f.weapon.damage * 1.18, 4.8, true);
                if (opponent.hp > 0) {
                  opponent.isGrounded = false;
                  opponent.velY = 13.6;
                  opponent.pos.y = Math.max(opponent.pos.y, opponent.groundY + 0.18);
                  opponent.stunTimer = 1.08;
                  opponent.launchedBySpecial = true;
                }
                this.spawnShockwave(
                  opponent.pos.x,
                  opponent.pos.y + 0.35,
                  opponent.pos.z,
                  3.2,
                  '#2DD4BF',
                );
                this.spawnSparks(opponent.pos.x, opponent.pos.y + 1.2, opponent.pos.z, 18, '#5EEAD4');
              }
            }
          }
          f.attackPhase = progress < 0.14 ? 'windup' : progress < 0.88 ? 'active' : 'recover';
        } else if (f.weapon.kind === 'lance') {
          // Achilles "Удар Грозы":
          // 0..68%: anime cut-in charge with glowing blue lightning orb building at spear tip
          // 68..77%: gameplay returns, enemy stands still, Achilles lowers/points spear at enemy
          // 77..92%: blue lightning launches from spear tip across arena into standing enemy
          f.rig.bladeTipPoint.getWorldPosition(_v1);
          if (_v1.y < f.pos.y + 0.25) {
            _v1.set(f.pos.x + Math.sin(f.yaw) * 1.1, f.pos.y + 1.45, f.pos.z + Math.cos(f.yaw) * 1.1);
          }
          this.thunderAuraGroup.position.copy(f.pos);
          if (!this.thunderShotActive && progress < 0.78) {
            this.thunderSpearOrb.position.copy(_v1);
            this.thunderSpearOrb.visible = true;
            this.thunderLight.position.set(_v1.x, _v1.y + 0.35, _v1.z);
          }

          if (progress < 0.77 && Math.random() < 0.56) {
            this.spawnSparks(_v1.x, _v1.y, _v1.z, 2, Math.random() > 0.25 ? '#38BDF8' : '#60A5FA');
          }

          if (f.specialStage === 0 && progress >= 0.03) {
            f.specialStage = 1;
            this.spawnLightningBolt(f.pos.x + 1.2, f.pos.y + 13.5, f.pos.z - 0.7, _v1.x, _v1.y, _v1.z, 1.25, '#38BDF8');
            this.spawnSparks(_v1.x, _v1.y, _v1.z, 12, '#60A5FA');
            this.thunderFlashIntensity = 15;
            this.cameraShake = Math.min(0.45, this.cameraShake + 0.07);
          }
          if (f.specialStage === 1 && progress >= 0.17) {
            f.specialStage = 2;
            this.spawnLightningBolt(f.pos.x - 1.4, f.pos.y + 14.8, f.pos.z - 0.8, _v1.x, _v1.y, _v1.z, 1.35, '#93C5FD');
            this.spawnShockwave(f.pos.x, f.pos.y + 0.16, f.pos.z, 3.0, '#38BDF8');
            this.spawnSparks(_v1.x, _v1.y, _v1.z, 14, '#38BDF8');
            this.thunderFlashIntensity = 17;
            this.cameraShake = Math.min(0.45, this.cameraShake + 0.08);
          }
          if (f.specialStage === 2 && progress >= 0.28) {
            f.specialStage = 3;
            this.spawnLightningBolt(f.pos.x - 0.7, f.pos.y + 14.0, f.pos.z + 0.8, _v1.x, _v1.y, _v1.z, 1.2, '#60A5FA');
            this.spawnSparks(_v1.x, _v1.y, _v1.z, 16, '#93C5FD');
            soundEngine.playThunderCharge();
            this.thunderFlashIntensity = 20;
            this.cameraShake = Math.min(0.45, this.cameraShake + 0.09);
          }
          if (f.specialStage === 3 && progress >= 0.45) {
            f.specialStage = 4;
            for (let k = 0; k < 3; k++) {
              const ang = (k / 3) * Math.PI * 2 + 0.4;
              this.spawnLightningBolt(
                _v1.x + Math.cos(ang) * 2.1,
                f.pos.y + 16.0,
                _v1.z + Math.sin(ang) * 2.1,
                _v1.x,
                _v1.y,
                _v1.z,
                1.35,
                k === 0 ? '#E0F2FE' : '#38BDF8',
              );
            }
            this.spawnShockwave(f.pos.x, f.pos.y + 0.24, f.pos.z, 4.1, '#38BDF8');
            this.spawnSparks(_v1.x, _v1.y, _v1.z, 22, '#60A5FA');
            soundEngine.playThunderCharge();
            this.thunderFlashIntensity = 22;
            this.cameraShake = Math.min(0.45, this.cameraShake + 0.1);
          }
          if (f.specialStage === 4 && progress >= 0.62) {
            f.specialStage = 5;
            for (let k = 0; k < 4; k++) {
              const ang = (k / 4) * Math.PI * 2;
              this.spawnLightningBolt(
                _v1.x + Math.cos(ang) * 2.4,
                f.pos.y + 16.5,
                _v1.z + Math.sin(ang) * 2.4,
                _v1.x,
                _v1.y,
                _v1.z,
                1.5,
                k % 2 === 0 ? '#E0F2FE' : '#2563EB',
              );
            }
            this.spawnShockwave(f.pos.x, f.pos.y + 0.28, f.pos.z, 4.6, '#60A5FA');
            this.spawnSparks(_v1.x, _v1.y, _v1.z, 24, '#38BDF8');
            soundEngine.playThunderCharge();
            this.thunderFlashIntensity = 24;
            this.cameraShake = Math.min(0.45, this.cameraShake + 0.12);
          }

          if (progress >= 0.68 && f.rig.specialAnimActive) {
            // Cut-in finished: transition smoothly into gameplay spear-pointing aim pose
            stopCustomSpecialClip(f.rig);
          }

          if (progress < 0.68) {
            f.attackPhase = 'windup';
          } else if (progress < 0.91) {
            f.attackPhase = 'active';
            if (f.specialStage === 5 && progress >= 0.77 && !f.hasHitCurrentSwing) {
              f.specialStage = 6;
              // Launch charged blue lightning from spear tip toward standing opponent!
              this.thunderShotActive = true;
              this.thunderShotTimer = 0;
              this.thunderShotOrigin.copy(_v1);
              this.thunderShotPos.copy(_v1);
              const shotDist = Math.hypot(
                opponent.pos.x - _v1.x,
                opponent.pos.z - _v1.z,
              );
              this.thunderShotDuration = THREE.MathUtils.clamp(shotDist / 30, 0.24, 0.44);
              this.thunderShotCaster = f;
              this.thunderShotTarget = opponent;
              this.thunderShotArcCooldown = 0;
              this.thunderFlashIntensity = 24;
              this.cameraShake = Math.min(0.45, this.cameraShake + 0.16);
              soundEngine.playThunderCharge();
              soundEngine.playLaser(true);
            } else if (progress >= 0.89 && !f.hasHitCurrentSwing && !this.thunderShotActive) {
              f.hasHitCurrentSwing = true;
              this.executeAttackImpact(f, opponent);
            }
          } else {
            f.attackPhase = 'recover';
          }
        } else if (f.weapon.kind === 'hammer') {
          // Xenon "Удар Ксенона": power coil -> soaring meteor leap -> earth-shatter hammer slam at 54% progress
          if (progress < 0.54) {
            f.attackPhase = 'windup';
          } else if (progress < 0.8) {
            f.attackPhase = 'active';
            if (!f.hasHitCurrentSwing) {
              f.hasHitCurrentSwing = true;
              this.executeAttackImpact(f, opponent);
            }
          } else {
            f.attackPhase = 'recover';
          }
        } else {
          if (progress < 0.28) {
            f.attackPhase = 'windup';
          } else if (progress < 0.65) {
            f.attackPhase = 'active';
            if (!f.hasHitCurrentSwing) {
              f.hasHitCurrentSwing = true;
              this.executeAttackImpact(f, opponent);
            }
          } else {
            f.attackPhase = 'recover';
          }
        }
      } else {
        if (progress < 0.28) {
          f.attackPhase = 'windup';
        } else if (progress < 0.65) {
          f.attackPhase = 'active';
          if (!f.hasHitCurrentSwing) {
            f.hasHitCurrentSwing = true;
            this.executeAttackImpact(f, opponent);
          }
        } else {
          f.attackPhase = 'recover';
        }
      }

      if (f.attackTimer <= 0) {
        f.attackPhase = 'none';
        f.isSpecialAttack = false;
        f.airAttacking = false;
        f.specialStage = 0;
        stopCustomSpecialClip(f.rig);
      }
    }

    let desiredVx = 0;
    let desiredVz = 0;

    if (f.isDashing) {
      f.dashTimer -= dt;
      desiredVx = f.dashDir.x * f.stats.dashSpeed;
      desiredVz = f.dashDir.z * f.stats.dashSpeed;
      if (f.isPlayer && !this.lockOnActive) {
        f.targetYaw = this.getCamForwardYaw();
      } else {
        f.targetYaw = Math.atan2(f.dashDir.x, f.dashDir.z);
      }
      if (f.dashTimer <= 0) {
        f.isDashing = false;
      }
    } else if (f.stunTimer > 0) {
      desiredVx = 0;
      desiredVz = 0;
    } else {
      const speedMult =
        f.attackPhase !== 'none'
          ? f.airAttacking
            ? 0.62
            : f.isAiming
              ? 0.45
              : 0.28
          : f.isGuarding
            ? 0.42
            : f.isAiming
              ? 0.52
              : !f.isGrounded
                ? 0.92
                : 1.0;
      desiredVx = moveInput.x * f.stats.moveSpeed * speedMult;
      desiredVz = moveInput.z * f.stats.moveSpeed * speedMult;

      const moveLen = Math.hypot(moveInput.x, moveInput.z);
      if (f.isSpecialAttack && f.weapon.kind === 'lance') {
        // Point Achilles straight toward opponent during "Удар Грозы" aim & lightning discharge
        f.targetYaw = Math.atan2(opponent.pos.x - f.pos.x, opponent.pos.z - f.pos.z);
      } else if (f.isPlayer && !this.lockOnActive) {
        // Player character turns strictly with the cursor/camera, never rotating toward WASD keys
        f.targetYaw =
          f.attackPhase !== 'none'
            ? (aimAngle ?? this.getCamForwardYaw())
            : this.getCamForwardYaw();
      } else if (aimAngle !== null && (this.lockOnActive || !f.isPlayer || f.isGuarding || f.isAiming)) {
        f.targetYaw = aimAngle;
      } else if (moveLen > 0.1) {
        f.targetYaw = Math.atan2(moveInput.x, moveInput.z);
      }
    }

    const accelLambda = f.isDashing
      ? 22
      : f.stunTimer > 0
        ? 9
        : !f.isGrounded
          ? 7.5
          : 12.5;
    f.vel.x = THREE.MathUtils.damp(f.vel.x, desiredVx, accelLambda, dt);
    f.vel.z = THREE.MathUtils.damp(f.vel.z, desiredVz, accelLambda, dt);

    f.pos.x += f.vel.x * dt;
    f.pos.z += f.vel.z * dt;

    const supportY = this.resolveCollisions(f, opponent);
    f.groundY = supportY;

    const grav = f.isDashing
      ? 6.5
      : f.airAttacking && !f.weapon.isRanged && f.attackPhase === 'windup'
        ? 10.5
        : 24.5;

    if (!f.isGrounded || f.pos.y > supportY || f.velY > 0) {
      f.velY -= grav * dt;
      f.pos.y += f.velY * dt;
      if (f.pos.y <= supportY) {
        const impactSpeed = Math.abs(f.velY);
        f.pos.y = supportY;
        f.velY = 0;
        if (!f.isGrounded) {
          f.isGrounded = true;
          f.jumpSquashTimer = Math.min(0.2, 0.08 + impactSpeed * 0.012);
          if (impactSpeed > 4.2) {
            soundEngine.playLand();
          }
          if (f.launchedBySpecial) {
            f.launchedBySpecial = false;
            this.spawnShockwave(f.pos.x, supportY + 0.14, f.pos.z, 3.2, '#2DD4BF');
            this.spawnSparks(f.pos.x, supportY + 0.25, f.pos.z, 14, '#5EEAD4');
            this.cameraShake = Math.min(0.45, this.cameraShake + 0.24);
            soundEngine.playHit(true);
          } else if (f.airAttacking && !f.weapon.isRanged && f.attackPhase !== 'none' && impactSpeed > 6.5) {
            this.spawnShockwave(f.pos.x, supportY + 0.12, f.pos.z, 2.35, f.weapon.accentHex);
            this.spawnSparks(f.pos.x, supportY + 0.25, f.pos.z, 9, f.weapon.accentHex);
          }
        }
      } else {
        f.isGrounded = false;
      }
    } else {
      f.pos.y = supportY;
      f.velY = 0;
      f.isGrounded = true;
    }

    const yawLambda = f.isPlayer && !this.lockOnActive ? 11.5 : 14;
    f.yaw = dampAngle(f.yaw, f.targetYaw, yawLambda, dt);
    f.rig.root.position.copy(f.pos);
    f.rig.root.rotation.y = f.yaw;

    // Anchor ground contact shadow on surface below while jumping
    const heightAboveSurface = Math.max(0, f.pos.y - f.groundY);
    f.rig.shadowDisc.position.y = f.groundY + 0.015 - f.pos.y;
    f.rig.shadowDisc.scale.setScalar(Math.max(0.45, 1 - heightAboveSurface * 0.2));
    (f.rig.shadowDisc.material as THREE.MeshBasicMaterial).opacity = Math.max(
      0.16,
      0.65 - heightAboveSurface * 0.16,
    );

    // Upper torso & head dynamically track cursor direction (quick waist/head lead + vertical pitch)
    let targetAimYaw = 0;
    let targetAimPitch = 0;
    if (f.isPlayer && !this.lockOnActive && f.hp > 0 && f.stunTimer <= 0 && !f.isDashing) {
      const camYaw = this.getCamForwardYaw();
      const rawDelta = shortestAngleDiff(f.yaw, camYaw);
      targetAimYaw = THREE.MathUtils.clamp(rawDelta * 1.15, -0.85, 0.85);
      targetAimPitch = THREE.MathUtils.clamp((this.camOrbitPitch - 0.36) * 0.72, -0.34, 0.34);
    }
    f.aimTorsoYaw = THREE.MathUtils.damp(f.aimTorsoYaw, targetAimYaw, 20, dt);
    f.aimTorsoPitch = THREE.MathUtils.damp(f.aimTorsoPitch, targetAimPitch, 18, dt);

    this.animateFighterRig(f, dt);
  }

  private executeAttackImpact(f: FighterRuntime, target: FighterRuntime) {
    if (f.isSpecialAttack) {
      if (f.weapon.isRanged) {
        [-0.1, 0, 0.1].forEach((spread) => {
          this.spawnProjectile(
            f,
            target,
            f.weapon.damage * 0.92,
            38,
            true,
            spread,
          );
        });
        soundEngine.playLaser(true);
      } else if (f.weapon.kind === 'lance') {
        // Achilles "Удар Грозы" blue lightning detonation on standing enemy
        const strikeX = target.pos.x;
        const strikeY = target.pos.y;
        const strikeZ = target.pos.z;
        f.rig.bladeTipPoint.getWorldPosition(_v1);
        if (_v1.y < f.pos.y + 0.3) {
          _v1.set(f.pos.x + Math.sin(f.yaw) * 1.1, f.pos.y + 1.35, f.pos.z + Math.cos(f.yaw) * 1.1);
        }
        // Main blue-white sky-to-enemy lightning pillar + 5 surrounding blue storm bolts
        this.spawnLightningBolt(
          strikeX,
          strikeY + 18.0,
          strikeZ,
          strikeX,
          strikeY + 0.15,
          strikeZ,
          2.1,
          '#E0F2FE',
        );
        for (let b = 0; b < 5; b++) {
          const ang = (b / 5) * Math.PI * 2 + 0.3;
          const rad = 1.25 + (b % 2) * 0.95;
          const bx = strikeX + Math.cos(ang) * rad;
          const bz = strikeZ + Math.sin(ang) * rad;
          this.spawnLightningBolt(
            bx + Math.cos(ang) * 1.1,
            strikeY + 15.5,
            bz + Math.sin(ang) * 1.1,
            bx,
            strikeY + 0.12,
            bz,
            1.25,
            b % 2 === 0 ? '#38BDF8' : '#60A5FA',
          );
        }
        // Final spear-tip -> enemy lightning arc
        this.spawnLightningBolt(
          _v1.x,
          _v1.y,
          _v1.z,
          strikeX,
          strikeY + 1.08,
          strikeZ,
          1.6,
          '#93C5FD',
        );

        this.spawnShockwave(strikeX, strikeY + 0.18, strikeZ, f.weapon.range + 2.8, '#E0F2FE');
        this.spawnShockwave(strikeX, strikeY + 0.25, strikeZ, f.weapon.range + 2.1, '#38BDF8');
        this.spawnShockwave(strikeX, strikeY + 0.32, strikeZ, f.weapon.range + 1.2, '#2563EB');
        this.spawnSparks(strikeX, strikeY + 1.05, strikeZ, 38, '#60A5FA');
        this.thunderLight.position.set(strikeX, strikeY + 2.1, strikeZ);
        this.thunderFlashIntensity = 30;
        this.cameraShake = Math.min(0.45, this.cameraShake + 0.44);
        soundEngine.playHit(true);
        soundEngine.playThunderStrike();

        for (let i = 0; i < this.obstacleStates.length; i++) {
          const obs = this.obstacleStates[i];
          if (obs.destroyed || !obs.data.destructible) continue;
          const odist = Math.hypot(obs.data.x - strikeX, obs.data.z - strikeZ);
          if (odist <= f.weapon.range + 2.2) {
            this.damageDestructibleObstacle(obs, f.weapon.damage * 2.0, f);
          }
        }

        if (target.hp > 0) {
          this.applyHit(f, target, f.weapon.damage * 2.15, 12.6, true);
        }
      } else if (f.weapon.kind === 'hammer') {
        // Xenon "Удар Ксенона": cataclysmic earth-shatter hammer impact
        const fx = Math.sin(f.yaw);
        const fz = Math.cos(f.yaw);
        f.vel.x += fx * 12.5;
        f.vel.z += fz * 12.5;
        if (!f.isGrounded) {
          f.velY = Math.min(f.velY, -13.0);
        }
        const strikeX = f.pos.x + fx * 1.75;
        const strikeZ = f.pos.z + fz * 1.75;
        this.spawnShockwave(strikeX, f.pos.y + 0.16, strikeZ, f.weapon.range + 2.1, '#F59E0B');
        this.spawnShockwave(strikeX, f.pos.y + 0.26, strikeZ, f.weapon.range + 1.0, '#EF4444');
        this.spawnSparks(strikeX, f.pos.y + 0.45, strikeZ, 26, '#FBBF24');
        this.cameraShake = Math.min(0.45, this.cameraShake + 0.36);
        soundEngine.playHit(true);

        for (let i = 0; i < this.obstacleStates.length; i++) {
          const obs = this.obstacleStates[i];
          if (obs.destroyed || !obs.data.destructible) continue;
          const odist = Math.hypot(obs.data.x - strikeX, obs.data.z - strikeZ);
          if (odist <= f.weapon.range + 1.8) {
            this.damageDestructibleObstacle(obs, f.weapon.damage * 2.0, f);
          }
        }

        const dist = Math.hypot(target.pos.x - f.pos.x, target.pos.z - f.pos.z);
        if (dist <= f.weapon.range + 2.15 && Math.abs(target.pos.y - f.pos.y) <= 2.7) {
          this.applyHit(f, target, f.weapon.damage * 2.18, 13.0, true);
        }
      } else {
        const fx = Math.sin(f.yaw);
        const fz = Math.cos(f.yaw);
        f.vel.x += fx * 12;
        f.vel.z += fz * 12;
        if (!f.isGrounded) {
          f.velY = Math.min(f.velY, -10.5);
        }
        this.spawnShockwave(
          f.pos.x + fx * 1.4,
          f.pos.y + 0.25,
          f.pos.z + fz * 1.4,
          f.weapon.range + 1.4,
          f.weapon.accentHex,
        );
        const dist = Math.hypot(target.pos.x - f.pos.x, target.pos.z - f.pos.z);
        if (dist <= f.weapon.range + 1.8 && Math.abs(target.pos.y - f.pos.y) <= 2.5) {
          this.applyHit(f, target, f.weapon.damage * 1.95, 11, true);
        }
      }
      return;
    }

    if (f.weapon.isRanged) {
      const shotSpeed = f.isAiming ? 42.0 : 36.0;
      const shotDamage = f.isAiming ? f.weapon.damage * 1.15 : f.weapon.damage;
      this.spawnProjectile(f, target, shotDamage, shotSpeed, false, 0);
      f.vel.x -= Math.sin(f.yaw) * (f.isGrounded ? (f.isAiming ? 1.0 : 1.6) : 2.4);
      f.vel.z -= Math.cos(f.yaw) * (f.isGrounded ? (f.isAiming ? 1.0 : 1.6) : 2.4);
    } else {
      const fx = Math.sin(f.yaw);
      const fz = Math.cos(f.yaw);
      const lungeImpulse = f.airAttacking ? 6.8 : f.comboStep === 3 ? 6.5 : 4.2;
      f.vel.x += fx * lungeImpulse;
      f.vel.z += fz * lungeImpulse;
      if (f.airAttacking) {
        f.velY = Math.min(f.velY, -10.0);
      }

      // Check destructible crates in melee swing arc!
      for (let i = 0; i < this.obstacleStates.length; i++) {
        const obs = this.obstacleStates[i];
        if (obs.destroyed || !obs.data.destructible) continue;
        const odx = obs.data.x - f.pos.x;
        const odz = obs.data.z - f.pos.z;
        const odist = Math.hypot(odx, odz);
        const dot = odist > 0.01 ? (fx * odx + fz * odz) / odist : 1;
        if (
          odist <= f.weapon.range + obs.data.radius * 0.6 &&
          dot > 0.05 &&
          f.pos.y <= obs.data.height + 1.5
        ) {
          this.damageDestructibleObstacle(obs, f.weapon.damage * 0.9, f);
        }
      }

      const dx = target.pos.x - f.pos.x;
      const dz = target.pos.z - f.pos.z;
      const dist = Math.hypot(dx, dz);
      const dot = dist > 0.01 ? (fx * dx + fz * dz) / dist : 1;
      const dyMax = f.airAttacking ? 2.45 : 1.75;

      if (dist <= f.weapon.range && dot > 0.15 && Math.abs(target.pos.y - f.pos.y) <= dyMax) {
        const mult = f.airAttacking
          ? 1.24
          : f.comboStep === 3
            ? 1.38
            : f.comboStep === 2
              ? 1.12
              : 1.0;
        const kb = f.airAttacking ? 7.8 : f.comboStep === 3 ? 8.5 : 4.5;
        this.applyHit(
          f,
          target,
          f.weapon.damage * mult,
          kb,
          f.comboStep === 3 || f.airAttacking,
        );
      }
    }
  }

  private resolveCollisions(f: FighterRuntime, other: FighterRuntime): number {
    const botRadius = 0.68;
    let supportGroundY = 0;

    for (let i = 0; i < this.obstacleStates.length; i++) {
      const obs = this.obstacleStates[i];
      if (obs.destroyed) continue;
      const dx = f.pos.x - obs.data.x;
      const dz = f.pos.z - obs.data.z;
      const dist = Math.hypot(dx, dz);
      const topSupportR = obs.data.radius * 0.96 + botRadius * 0.55;
      const minD = obs.data.radius + botRadius;
      if (dist < topSupportR && f.pos.y >= obs.data.height - 0.45 && f.velY <= 1.2) {
        supportGroundY = Math.max(supportGroundY, obs.data.height);
      } else if (dist < minD && dist > 0.0001) {
        const clearingTopLedge = !f.isGrounded && f.pos.y >= obs.data.height - 0.55 && f.velY > -2.5;
        if (!clearingTopLedge && f.pos.y < obs.data.height - 0.45) {
          const push = minD - dist;
          f.pos.x += (dx / dist) * push;
          f.pos.z += (dz / dist) * push;
        }
      }
    }

    const fdx = f.pos.x - other.pos.x;
    const fdz = f.pos.z - other.pos.z;
    const fdist = Math.hypot(fdx, fdz);
    const minFighterDist = 1.25;
    if (fdist < minFighterDist && fdist > 0.0001 && Math.abs(f.pos.y - other.pos.y) < 1.65) {
      const push = (minFighterDist - fdist) * 0.5;
      f.pos.x += (fdx / fdist) * push;
      f.pos.z += (fdz / fdist) * push;
    }

    const r = Math.hypot(f.pos.x, f.pos.z);
    const maxR = this.arena.radius - botRadius;
    if (r > maxR && r > 0.0001) {
      f.pos.x = (f.pos.x / r) * maxR;
      f.pos.z = (f.pos.z / r) * maxR;
    }

    return supportGroundY;
  }

  private animateFighterRig(f: FighterRuntime, dt: number) {
    const rig = f.rig;
    rig.mixer?.update(dt);
    if (rig.specialAnimActive && rig.specialMixer) {
      rig.specialMixer.update(dt);
    }
    const speed = Math.hypot(f.vel.x, f.vel.z);
    const speedRatio = THREE.MathUtils.clamp(speed / f.stats.moveSpeed, 0, 1.8);
    const forwardDot = Math.sin(f.yaw) * f.vel.x + Math.cos(f.yaw) * f.vel.z;
    const rightDot = -Math.cos(f.yaw) * f.vel.x + Math.sin(f.yaw) * f.vel.z;
    const strafeRatio = THREE.MathUtils.clamp(rightDot / f.stats.moveSpeed, -1.2, 1.2);
    const strideSign = forwardDot < -0.4 ? -1 : 1;

    const dashDuration = f.stats.dashDuration;
    const dashProgress = f.isDashing
      ? THREE.MathUtils.clamp(1 - f.dashTimer / dashDuration, 0, 1)
      : 0;
    const dashForward = f.isDashing
      ? Math.sin(f.yaw) * f.dashDir.x + Math.cos(f.yaw) * f.dashDir.z
      : 1;
    const dashRight = f.isDashing
      ? -Math.cos(f.yaw) * f.dashDir.x + Math.sin(f.yaw) * f.dashDir.z
      : 0;

    f.stridePhase += dt * (2.6 + speedRatio * 9.5) * strideSign;
    const t = this.elapsedTime;
    const breath = rig.isCustomModel
      ? 0
      : Math.sin(t * 2.6 + (f.isPlayer ? 0 : 1.5)) * 0.028;

    const sMove = Math.min(1.4, speedRatio);
    const sprint = THREE.MathUtils.clamp((sMove - 0.52) / 0.68, 0, 1);
    const absStrafe = Math.min(1, Math.abs(strafeRatio));
    const sinP = Math.sin(f.stridePhase);
    const cosP = Math.cos(f.stridePhase);
    const recL = Math.pow(Math.max(0, Math.sin(f.stridePhase * strideSign - 0.62)), 0.76);
    const recR = Math.pow(Math.max(0, -Math.sin(f.stridePhase * strideSign - 0.62)), 0.76);
    const plantL = Math.pow(Math.max(0, -sinP * strideSign), 1.12);
    const plantR = Math.pow(Math.max(0, sinP * strideSign), 1.12);
    const swingAmp = (0.5 + 0.26 * sprint) * (1 - absStrafe * 0.22);

    const leanMult =
      f.weapon.kind === 'kodachi'
        ? 1.38
        : f.weapon.kind === 'hammer'
          ? 1.08
          : f.weapon.isRanged
            ? 0.88
            : 1.0;

    let bodyPitch =
      (forwardDot < -0.4 ? -0.11 : 0.13 + 0.15 * sprint) * Math.min(1, sMove) * leanMult +
      Math.sin(f.stridePhase * 2) * 0.025 * sMove;
    let bodyRoll = strafeRatio * (0.15 + 0.07 * sprint) - sinP * 0.042 * sMove;
    let bodyBob =
      breath + (-Math.cos(f.stridePhase * 2 + 0.35) * (0.048 + 0.014 * sprint) + 0.012) * sMove;
    let torsoYaw = cosP * (0.14 + 0.08 * sprint) * sMove + strafeRatio * 0.09;
    let torsoPitch =
      f.weapon.kind === 'kodachi' ? 0.16 * sMove : f.weapon.kind === 'hammer' ? 0.08 * sMove : 0;

    let lHipX = (sinP * swingAmp + Math.sin(f.stridePhase * 2) * 0.04) * sMove;
    let rHipX = (-sinP * swingAmp + Math.sin(f.stridePhase * 2) * 0.04) * sMove;
    let lHipZ = 0.05 + Math.max(0, -strafeRatio) * 0.13 + absStrafe * 0.03;
    let rHipZ = -0.05 - Math.max(0, strafeRatio) * 0.13 - absStrafe * 0.03;
    let lKneeX = (recL * (0.78 + 0.36 * sprint) + plantL * 0.28) * sMove + 0.12;
    let rKneeX = (recR * (0.78 + 0.36 * sprint) + plantR * 0.28) * sMove + 0.12;

    let lShoulderX = -sinP * (0.44 + 0.16 * sprint) * sMove - 0.16;
    let lShoulderY = 0;
    let lShoulderZ = 0.18;
    let lElbowX = -0.48 - recL * 0.16 * sMove;

    let rShoulderX = sinP * (0.38 + 0.14 * sprint) * sMove - 0.24;
    let rShoulderZ = -0.18;
    let rElbowX = -0.64 - recR * 0.14 * sMove;

    let headYaw = -torsoYaw * 0.7 - strafeRatio * 0.05;

    if (f.weapon.isRanged) {
      // Tactical bladed two-handed rifle carry & shouldered aim
      torsoYaw = -0.45 + sinP * 0.045 * sMove;
      torsoPitch = 0.07;
      headYaw = 0.42 - sinP * 0.035 * sMove;
      rShoulderX = -1.16 - sMove * 0.06 + Math.cos(f.stridePhase * 2) * 0.04 * sMove;
      rShoulderZ = -0.14;
      rElbowX = -0.34;
      lShoulderX = -1.06 + Math.cos(f.stridePhase * 2) * 0.035 * sMove;
      lShoulderZ = -0.56;
      lElbowX = -0.56;
    } else if (f.weapon.kind === 'kodachi') {
      // Kunoichi agile shinobi sprint with swept-back arms
      if (sMove > 0.15) {
        lShoulderX = 0.68 * Math.min(1.15, sMove) - sinP * 0.18 * sMove;
        rShoulderX = 0.68 * Math.min(1.15, sMove) + sinP * 0.18 * sMove;
        lShoulderZ = 0.34;
        rShoulderZ = -0.34;
        lElbowX = -0.34;
        rElbowX = -0.34;
      } else {
        lShoulderX = -0.78;
        lShoulderZ = 0.24;
        lElbowX = -1.12;
        rShoulderX = 0.22;
        rShoulderZ = -0.26;
        rElbowX = -0.92;
      }
    } else if (f.weapon.kind === 'hammer') {
      // Heavy Xenon warhammer poise: upright hammer held cleanly beside right shoulder, left fist poised at waist
      if (sMove < 0.15) {
        torsoYaw = -0.22;
        headYaw = 0.2;
        lHipX = -0.2;
        lKneeX = 0.32;
        rHipX = 0.2;
        rKneeX = 0.24;
        rShoulderX = -0.18 + breath * 0.6;
        rShoulderZ = 0.36;
        rElbowX = -1.04;
        lShoulderX = -0.16 + breath * 0.5;
        lShoulderZ = -0.28;
        lElbowX = -0.8;
      } else {
        torsoYaw = -0.2 + cosP * 0.08 * sMove;
        headYaw = 0.18 - cosP * 0.06 * sMove;
        rShoulderX = -0.32 + sinP * 0.16 * sMove;
        rShoulderZ = 0.32;
        rElbowX = -0.96;
        lShoulderX = -0.32 - sinP * 0.48 * sMove;
        lShoulderZ = -0.24;
        lElbowX = -0.8;
      }
    } else if (f.weapon.kind === 'lance') {
      // Achilles hoplite shield + spear counter-poise
      lShoulderX = -0.54 - sinP * 0.2 * sMove;
      lShoulderY = -0.58;
      lElbowX = -0.68;
      rShoulderX = 0.2 + sinP * 0.2 * sMove;
      rShoulderZ = -0.14;
      rElbowX = -0.9;
    }

    if (f.isDashing) {
      const snap = Math.min(1, dashProgress * 3.0);
      const snapEase = 1 - Math.pow(1 - snap, 4);
      const brake = dashProgress > 0.7 ? (dashProgress - 0.7) / 0.3 : 0;
      const strength = snapEase * (1 - brake * 0.45);
      const isLateral = Math.abs(dashRight) > Math.abs(dashForward) * 0.9 && Math.abs(dashRight) > 0.35;

      if (isLateral) {
        const side = dashRight >= 0 ? 1 : -1;
        bodyPitch = (0.16 + brake * 0.08) * strength;
        bodyRoll = side * 0.38 * strength;
        bodyBob = (0.06 - brake * 0.08) * strength;
        torsoYaw = (f.weapon.isRanged ? -0.42 : 0) + side * 0.28 * strength;
        headYaw = (f.weapon.isRanged ? 0.38 : 0) - side * 0.2 * strength;
        if (side > 0) {
          // Dashing RIGHT: right knee tucked, left leg extended push-off
          rHipX = -0.64 * strength;
          rKneeX = 1.08 * strength;
          lHipX = 0.34 * strength;
          lKneeX = 0.24 * strength;
          lHipZ = 0.2 * strength;
          rHipZ = -0.08;
        } else {
          // Dashing LEFT: left knee tucked, right leg extended push-off
          lHipX = -0.64 * strength;
          lKneeX = 1.08 * strength;
          rHipX = 0.34 * strength;
          rKneeX = 0.24 * strength;
          rHipZ = -0.2 * strength;
          lHipZ = 0.08;
        }
        if (f.weapon.isRanged) {
          rShoulderX = side > 0 ? -1.04 : -0.96;
          lShoulderX = side > 0 ? -1.08 : -1.16;
        } else if (f.weapon.kind === 'kodachi') {
          bodyRoll = side * 0.44 * strength;
          if (side > 0) {
            rShoulderX = -0.94;
            rShoulderZ = 0.38;
            rElbowX = -1.06;
            lShoulderX = 0.72 * strength;
            lShoulderZ = 0.48;
          } else {
            lShoulderX = -0.94;
            lShoulderZ = -0.38;
            lElbowX = -1.06;
            rShoulderX = 0.72 * strength;
            rShoulderZ = -0.48;
          }
        } else if (f.weapon.kind === 'hammer') {
          if (side > 0) {
            rShoulderX = -0.38;
            rShoulderZ = -0.58 * strength;
            lShoulderX = -0.98;
            lShoulderZ = -0.18;
          } else {
            rShoulderX = -0.78;
            rShoulderZ = 0.22 * strength;
            lShoulderX = -0.42;
            lShoulderZ = 0.56 * strength;
          }
        } else if (f.weapon.kind === 'lance') {
          if (side > 0) {
            lShoulderX = -0.92;
            lShoulderY = -1.08;
            rShoulderX = 0.66 * strength;
            rShoulderZ = -0.28;
          } else {
            lShoulderX = -1.04;
            lShoulderY = -1.24;
            lShoulderZ = 0.18;
            rShoulderX = 0.38;
          }
        } else {
          if (side > 0) {
            rShoulderX = 0.68 * strength;
            rShoulderZ = -0.36;
            lShoulderX = -0.82;
            lShoulderZ = 0.24;
          } else {
            lShoulderX = 0.68 * strength;
            lShoulderZ = 0.36;
            rShoulderX = -0.82;
            rShoulderZ = -0.24;
          }
        }
      } else if (dashForward < -0.3) {
        bodyPitch = -0.26 * strength;
        bodyRoll = dashRight * 0.18 * strength;
        bodyBob = 0.07 * strength;
        lHipX = -0.46 * strength;
        lKneeX = 0.58 * strength;
        rHipX = 0.24 * strength;
        rKneeX = 0.82 * strength;
        if (!f.weapon.isRanged) {
          lShoulderX = -0.72 * strength;
          rShoulderX = -0.62 * strength;
          lShoulderZ = 0.32;
          rShoulderZ = -0.32;
        }
      } else {
        bodyPitch = 0.48 * strength;
        bodyRoll = dashRight * 0.18 * strength;
        bodyBob = 0.11 * strength;
        lHipX = 0.6 * strength;
        lKneeX = 0.42 * strength;
        rHipX = 0.2 * strength;
        rKneeX = 0.94 * strength;
        if (!f.weapon.isRanged) {
          lShoulderX = 0.72 * strength;
          rShoulderX = 0.72 * strength;
          lShoulderZ = 0.28;
          rShoulderZ = -0.28;
        }
      }
    } else if (f.isGuarding) {
      bodyPitch = 0.04;
      bodyRoll = 0;
      bodyBob = -0.05;
      lHipX = -0.24;
      rHipX = 0.24;
      lKneeX = 0.34;
      rKneeX = 0.34;
      if (f.weapon.kind === 'lance') {
        torsoYaw = 0.35;
        headYaw = -0.32;
        lShoulderX = -1.18;
        lShoulderY = -0.85;
        lShoulderZ = 0.2;
        lElbowX = -0.82;
        rShoulderX = -0.35;
        rShoulderZ = 0.15;
        rElbowX = -0.95;
      } else {
        lShoulderX = -1.05;
        lShoulderZ = -0.35;
        lElbowX = -0.85;
        rShoulderX = -0.85;
        rShoulderZ = 0.25;
        rElbowX = -0.95;
      }
    } else if (f.stunTimer > 0) {
      bodyPitch = -0.28;
      bodyRoll = 0;
      torsoPitch = -0.22;
      lShoulderX = -0.4;
      rShoulderX = -0.4;
      lShoulderZ = 0.55;
      rShoulderZ = -0.55;
    } else if (!f.isGrounded && f.attackPhase === 'none') {
      // Multi-phase airborne jump: launch extension -> mid-ascent knee drive -> apex hang -> landing brace
      const rise = THREE.MathUtils.clamp(f.velY / 7.5, -1, 1);
      const launchLift = Math.max(0, rise);
      const fallBrace = Math.max(0, -rise);
      const hoverBreath = Math.sin(t * 5.2 + (f.isPlayer ? 0 : 1.1)) * 0.024;
      bodyRoll = strafeRatio * 0.16;
      bodyBob = 0.03 + hoverBreath * 0.6;
      lHipZ = 0.08 + Math.max(0, -strafeRatio) * 0.08;
      rHipZ = -0.08 - Math.max(0, strafeRatio) * 0.08;
      if (rise > 0.56) {
        const launch = THREE.MathUtils.clamp((rise - 0.56) / 0.44, 0, 1);
        bodyPitch = THREE.MathUtils.lerp(-0.15, -0.22, launch);
        lHipX = THREE.MathUtils.lerp(-0.96, -1.04, launch);
        lKneeX = THREE.MathUtils.lerp(1.36, 1.44, launch);
        rHipX = THREE.MathUtils.lerp(0.42, 0.58, launch);
        rKneeX = THREE.MathUtils.lerp(0.32, 0.18, launch);
      } else if (rise > 0.18) {
        const a = THREE.MathUtils.clamp((rise - 0.18) / 0.38, 0, 1);
        bodyPitch = THREE.MathUtils.lerp(0.02, -0.15, a);
        lHipX = THREE.MathUtils.lerp(-0.7, -0.96, a);
        lKneeX = THREE.MathUtils.lerp(1.08, 1.36, a);
        rHipX = THREE.MathUtils.lerp(0.28, 0.42, a);
        rKneeX = THREE.MathUtils.lerp(0.64, 0.32, a);
      } else if (rise < -0.2) {
        const d = THREE.MathUtils.clamp((-rise - 0.2) / 0.8, 0, 1);
        bodyPitch = THREE.MathUtils.lerp(0.02, 0.19, d);
        lHipX = THREE.MathUtils.lerp(-0.7, -0.32, d);
        lKneeX = THREE.MathUtils.lerp(1.08, 0.52, d);
        rHipX = THREE.MathUtils.lerp(0.28, 0.14, d);
        rKneeX = THREE.MathUtils.lerp(0.64, 0.44, d);
      } else {
        bodyPitch = 0.02 + hoverBreath;
        lHipX = -0.7 + hoverBreath * 0.4;
        lKneeX = 1.08;
        rHipX = 0.28 - hoverBreath * 0.4;
        rKneeX = 0.64;
      }
      if (!f.weapon.isRanged) {
        if (f.weapon.kind === 'kodachi') {
          lShoulderX = 0.42 - launchLift * 0.52 + fallBrace * 0.25;
          lShoulderZ = 0.46 + fallBrace * 0.18;
          lElbowX = -0.68;
          rShoulderX = -0.76 - launchLift * 0.28 + fallBrace * 0.2;
          rShoulderZ = -0.32 - fallBrace * 0.16;
          rElbowX = -1.02;
        } else {
          lShoulderX = -0.66 - launchLift * 0.28 + fallBrace * 0.22;
          lShoulderZ = 0.28 + fallBrace * 0.18;
          lElbowX = -0.58;
          rShoulderX = -0.48 - launchLift * 0.26 + fallBrace * 0.18;
          rShoulderZ = -0.26 - fallBrace * 0.16;
          rElbowX = -0.66;
        }
      }
    } else if (f.attackPhase !== 'none') {
      const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
      bodyRoll = 0;
      if (f.weapon.isRanged) {
        // Horizontal shoulder-level sniper aim & crisp recoil kick (grounded or mid-air)
        const brace = p < 0.18 ? p / 0.18 : Math.max(0, 1 - (p - 0.18) / 0.22);
        const kick =
          p >= 0.18 && p <= 0.5
            ? Math.sin(((p - 0.18) / 0.32) * Math.PI) * (f.isSpecialAttack ? 0.36 : 0.24)
            : 0;
        if (!f.isGrounded) {
          bodyPitch = -0.06 - kick * 0.18;
          bodyBob = 0.03;
          lHipX = -0.68 + kick * 0.22;
          lKneeX = 1.04 - kick * 0.18;
          rHipX = 0.36 + kick * 0.14;
          rKneeX = 0.48;
        } else {
          bodyPitch = 0.04 * brace - kick * 0.12;
          bodyBob = -0.04 * brace - kick * 0.03;
          lKneeX = 0.34 + brace * 0.08 + kick * 0.06;
          rKneeX = 0.26 + kick * 0.14;
        }
        torsoYaw = -0.48 - kick * 0.12;
        torsoPitch = 0.06 - kick * 0.32;
        headYaw = 0.44 + kick * 0.06;
        rShoulderX = -1.2 - kick;
        rShoulderZ = -0.12;
        rElbowX = -0.3 - kick * 0.12;
        lShoulderX = -1.1 - kick * 0.92;
        lShoulderZ = -0.54;
        lElbowX = -0.52;
      } else if (!f.isGrounded || f.airAttacking) {
        // Aerial diving melee strike!
        const w = Math.min(1, p / 0.28);
        const a = THREE.MathUtils.clamp((p - 0.28) / 0.32, 0, 1);
        const isW = p < 0.28;
        bodyPitch = isW ? -0.16 * w : THREE.MathUtils.lerp(-0.16, 0.44, a);
        torsoPitch = isW ? -0.26 * w : THREE.MathUtils.lerp(-0.26, 0.28, a);
        torsoYaw = isW ? -0.35 * w : THREE.MathUtils.lerp(-0.35, 0.32, a);
        headYaw = -torsoYaw * 0.5;
        lHipX = isW ? -0.82 : -0.44;
        lKneeX = isW ? 1.22 : 0.46;
        rHipX = isW ? -0.22 : 0.56;
        rKneeX = isW ? 0.72 : 0.58;
        rShoulderX = isW ? -2.32 * w : THREE.MathUtils.lerp(-2.32, -0.78, a);
        rElbowX = isW ? -0.55 * w : -0.08;
        lShoulderX = -0.75;
        lShoulderY = f.weapon.kind === 'lance' ? -0.65 : 0;
        lElbowX = -0.52;
      } else if (f.weapon.kind === 'lance') {
        const w = Math.min(1, p / 0.26);
        const a = THREE.MathUtils.clamp((p - 0.26) / 0.26, 0, 1);
        const aEase = 1 - Math.pow(1 - a, 3);
        const isW = p < 0.26;
        if (f.isSpecialAttack) {
          if (p >= 0.66) {
            const aimIn = THREE.MathUtils.smoothstep(p, 0.66, 0.74);
            const recoil =
              p >= 0.77 && p <= 0.9
                ? Math.sin(((p - 0.77) / 0.13) * Math.PI)
                : 0;
            const rec = THREE.MathUtils.smoothstep(p, 0.89, 1.0);
            const k = aimIn * (1 - rec);
            torsoYaw = 0.34 * k;
            bodyPitch = (0.08 - recoil * 0.1) * k;
            torsoPitch = (0.06 - recoil * 0.08) * k;
            bodyBob = -0.06 * k;
            lHipX = -0.44 * k;
            lKneeX = 0.62 * k;
            rHipX = 0.38 * k;
            rShoulderX = THREE.MathUtils.lerp(-0.42, -1.58 - recoil * 0.16, k);
            rElbowX = THREE.MathUtils.lerp(-0.55, -0.06, k);
            lShoulderX = THREE.MathUtils.lerp(-0.78, -0.72, k);
            lShoulderY = -0.75 * k;
          } else {
            const w = THREE.MathUtils.smoothstep(p, 0, 0.66);
            torsoPitch = -0.26 * w;
            bodyPitch = -0.14 * w;
            bodyBob = 0.09 * w;
            lHipX = -0.24 * w;
            lKneeX = 0.42 * w;
            rHipX = 0.18 * w;
            rShoulderX = -2.42 * w;
            rElbowX = -0.68 * w;
            lShoulderX = -0.98 * w;
            lShoulderY = -0.65;
          }
        } else if (f.comboStep === 3) {
          torsoPitch = isW ? -0.24 * w : THREE.MathUtils.lerp(-0.24, 0.32, aEase);
          bodyPitch = isW ? -0.14 * w : THREE.MathUtils.lerp(-0.14, 0.34, aEase);
          bodyBob = isW ? 0.09 * w : -0.13 * aEase;
          lHipX = isW ? -0.24 * w : -0.64 * aEase;
          lKneeX = isW ? 0.42 * w : 0.9 * aEase;
          rHipX = isW ? 0.18 * w : 0.54 * aEase;
          rShoulderX = isW ? -2.38 * w : THREE.MathUtils.lerp(-2.38, -0.84, aEase);
          rElbowX = isW ? -0.75 * w : -0.06;
          lShoulderX = isW ? -0.96 * w : -0.24;
          lShoulderY = -0.65;
        } else if (f.comboStep === 2) {
          torsoYaw = isW ? -0.72 * w : THREE.MathUtils.lerp(-0.72, 0.66, aEase);
          bodyRoll = isW ? -0.08 * w : 0.08 * aEase;
          bodyBob = isW ? -0.08 * w : -0.06 * aEase;
          headYaw = -torsoYaw * 0.55;
          lHipX = isW ? -0.28 : -0.56 * aEase;
          lKneeX = isW ? 0.46 : 0.76 * aEase;
          rHipX = isW ? 0.28 : 0.48 * aEase;
          rShoulderX = isW ? 0.38 * w : THREE.MathUtils.lerp(0.38, -1.48, aEase);
          rShoulderZ = isW ? -0.48 * w : THREE.MathUtils.lerp(-0.48, 0.44, aEase);
          rElbowX = isW ? -1.12 * w : -0.08;
          lShoulderX = isW ? -0.92 * w : -0.22;
          lShoulderY = -0.55;
        } else {
          torsoYaw = isW ? -0.66 * w : THREE.MathUtils.lerp(-0.66, 0.52, aEase);
          bodyRoll = isW ? -0.07 * w : 0.06 * aEase;
          headYaw = -torsoYaw * 0.65;
          bodyPitch = isW ? -0.06 * w : 0.26 * aEase;
          bodyBob = isW ? -0.07 * w : -0.1 * aEase;
          lHipX = isW ? -0.2 : -0.62 * aEase;
          lKneeX = isW ? 0.34 : 0.86 * aEase;
          rHipX = isW ? 0.26 : 0.52 * aEase;
          rShoulderX = isW ? 0.58 * w : THREE.MathUtils.lerp(0.58, -1.52, aEase);
          rElbowX = isW ? -1.28 * w : -0.06;
          lShoulderX = isW ? -0.95 * w : -0.2;
          lShoulderY = isW ? -0.78 * w : -0.25;
        }
      } else if (f.weapon.kind === 'hammer') {
        if (f.isSpecialAttack) {
          // Xenon "Удар Ксенона": squat load -> soaring leap (hammer trails) -> apex lock -> cataclysmic smash
          if (p < 0.18) {
            const c = THREE.MathUtils.smoothstep(p, 0, 0.18);
            bodyPitch = 0.2 * c;
            torsoPitch = 0.22 * c;
            bodyBob = -0.15 * c;
            lHipX = -0.5 * c;
            lKneeX = 0.9 * c;
            rHipX = 0.4 * c;
            rShoulderX = 0.56 * c;
            lShoulderX = -0.78 * c;
          } else if (p < 0.47) {
            const u = 1 - Math.pow(1 - (p - 0.18) / 0.29, 3);
            bodyPitch = THREE.MathUtils.lerp(0.2, -0.2, u);
            torsoPitch = THREE.MathUtils.lerp(0.22, -0.32, u);
            bodyBob = THREE.MathUtils.lerp(-0.15, 0.2, u);
            lHipX = THREE.MathUtils.lerp(-0.5, -0.78, u);
            lKneeX = THREE.MathUtils.lerp(0.9, 1.15, u);
            rHipX = THREE.MathUtils.lerp(0.4, -0.26, u);
            rShoulderX = THREE.MathUtils.lerp(0.56, -2.56, u);
            lShoulderX = THREE.MathUtils.lerp(-0.78, -2.26, u);
            rElbowX = -0.26;
          } else if (p < 0.6) {
            const s = 1 - Math.pow(1 - (p - 0.47) / 0.13, 4);
            bodyPitch = THREE.MathUtils.lerp(-0.2, 0.3, s);
            torsoPitch = THREE.MathUtils.lerp(-0.32, 0.34, s);
            bodyBob = THREE.MathUtils.lerp(0.2, -0.17, s);
            lHipX = THREE.MathUtils.lerp(-0.78, -0.68, s);
            lKneeX = THREE.MathUtils.lerp(1.15, 0.98, s);
            rHipX = THREE.MathUtils.lerp(-0.26, 0.52, s);
            rShoulderX = THREE.MathUtils.lerp(-2.56, -0.96, s);
            lShoulderX = THREE.MathUtils.lerp(-2.26, 0.42, s);
            rElbowX = -0.08;
          } else {
            const rec = THREE.MathUtils.smoothstep(p, 0.72, 1.0);
            bodyPitch = THREE.MathUtils.lerp(0.3, 0.04, rec);
            torsoPitch = THREE.MathUtils.lerp(0.34, 0.05, rec);
            bodyBob = THREE.MathUtils.lerp(-0.17, 0, rec);
            lHipX = THREE.MathUtils.lerp(-0.68, -0.14, rec);
            lKneeX = THREE.MathUtils.lerp(0.98, 0.14, rec);
            rHipX = THREE.MathUtils.lerp(0.52, 0.14, rec);
            rShoulderX = THREE.MathUtils.lerp(-0.96, -0.52, rec);
            lShoulderX = THREE.MathUtils.lerp(0.42, -0.42, rec);
          }
        } else if (f.comboStep === 3) {
          // Heavy Overhead Titan Crush with weapon lag & heavy hoist recovery
          if (p < 0.28) {
            const w = THREE.MathUtils.smoothstep(p, 0, 0.28);
            bodyPitch = -0.14 * w;
            torsoPitch = -0.26 * w;
            bodyBob = 0.06 * w;
            lHipX = -0.26 * w;
            lKneeX = 0.46 * w;
            rHipX = 0.2 * w;
            rShoulderX = -2.44 * w;
            lShoulderX = -1.94 * w;
            rElbowX = -0.36;
          } else if (p < 0.56) {
            const a = 1 - Math.pow(1 - (p - 0.28) / 0.28, 4);
            bodyPitch = THREE.MathUtils.lerp(-0.14, 0.26, a);
            torsoPitch = THREE.MathUtils.lerp(-0.26, 0.28, a);
            bodyBob = THREE.MathUtils.lerp(0.06, -0.14, a);
            lHipX = THREE.MathUtils.lerp(-0.26, -0.64, a);
            lKneeX = THREE.MathUtils.lerp(0.46, 0.92, a);
            rHipX = THREE.MathUtils.lerp(0.2, 0.5, a);
            rShoulderX = THREE.MathUtils.lerp(-2.44, -0.96, a);
            lShoulderX = THREE.MathUtils.lerp(-1.94, 0.36, a);
            rElbowX = -0.1;
          } else {
            const r = THREE.MathUtils.smoothstep(p, 0.68, 1.0);
            bodyPitch = THREE.MathUtils.lerp(0.26, 0.04, r);
            torsoPitch = THREE.MathUtils.lerp(0.28, 0.04, r);
            bodyBob = THREE.MathUtils.lerp(-0.14, 0, r);
            lHipX = THREE.MathUtils.lerp(-0.64, -0.12, r);
            lKneeX = THREE.MathUtils.lerp(0.92, 0.14, r);
            rHipX = THREE.MathUtils.lerp(0.5, 0.12, r);
            rShoulderX = THREE.MathUtils.lerp(-0.96, -0.52, r);
            lShoulderX = THREE.MathUtils.lerp(0.36, -0.42, r);
          }
        } else {
          // Hip-first warhammer cleave with delayed arm whip
          const dir = f.comboStep === 2 ? -1 : 1;
          const hipPhase =
            p < 0.25
              ? -1.2 * dir * THREE.MathUtils.smoothstep(p, 0, 0.25)
              : p < 0.54
                ? THREE.MathUtils.lerp(-1.2 * dir, 1.28 * dir, 1 - Math.pow(1 - (p - 0.25) / 0.29, 3))
                : THREE.MathUtils.lerp(1.28 * dir, 0, THREE.MathUtils.smoothstep(p, 0.68, 1.0));
          const armLagP = THREE.MathUtils.clamp((p - 0.05) / 0.95, 0, 1);
          const armPhase =
            armLagP < 0.25
              ? -1.3 * dir * THREE.MathUtils.smoothstep(armLagP, 0, 0.25)
              : armLagP < 0.54
                ? THREE.MathUtils.lerp(-1.3 * dir, 1.32 * dir, 1 - Math.pow(1 - (armLagP - 0.25) / 0.29, 4))
                : THREE.MathUtils.lerp(1.32 * dir, 0, THREE.MathUtils.smoothstep(armLagP, 0.68, 1.0));
          bodyRoll = hipPhase * 0.09;
          bodyBob = p < 0.25 ? -0.06 : -0.1;
          lHipX = p < 0.25 ? -0.28 : -0.58;
          lKneeX = p < 0.25 ? 0.5 : 0.86;
          rHipX = p < 0.25 ? 0.28 : 0.46;
          torsoYaw = hipPhase * 0.72;
          headYaw = -torsoYaw * 0.68;
          torsoPitch = p < 0.25 ? -0.08 : 0.16;
          rShoulderX = -1.24;
          rShoulderZ = -0.28 + armPhase * 0.66;
          rElbowX = p < 0.25 ? -0.58 : -0.14;
          lShoulderX = -0.82;
          lShoulderZ = 0.24 - armPhase * 0.34;
        }
      } else if (f.isSpecialAttack && f.weapon.kind === 'kodachi') {
        if (p < 0.65) {
          const cycle = Math.sin(((p - 0.11) / 0.17) * Math.PI * 2);
          bodyBob = -0.09;
          bodyPitch = 0.24;
          bodyRoll = cycle * 0.14;
          lHipX = -0.5;
          lKneeX = 0.82;
          rHipX = 0.4;
          rKneeX = 0.62;
          torsoYaw = cycle * 0.94;
          headYaw = -torsoYaw * 0.72;
          rShoulderX = -1.32;
          rShoulderZ = -0.35 + cycle * 0.72;
          lShoulderX = -1.32;
          lShoulderZ = 0.35 - cycle * 0.72;
          rElbowX = -0.18;
        } else if (p < 0.76) {
          const c = THREE.MathUtils.smoothstep(p, 0.65, 0.76);
          bodyBob = THREE.MathUtils.lerp(-0.09, -0.16, c);
          bodyPitch = 0.28;
          torsoPitch = 0.28;
          lHipX = -0.58;
          lKneeX = 1.0;
          rShoulderX = THREE.MathUtils.lerp(-1.2, 0.84, c);
          lShoulderX = THREE.MathUtils.lerp(-1.2, 0.84, c);
        } else {
          const up = THREE.MathUtils.clamp((p - 0.76) / 0.14, 0, 1);
          const upEase = 1 - Math.pow(1 - up, 4);
          bodyBob = THREE.MathUtils.lerp(-0.16, 0.22, upEase);
          bodyPitch = THREE.MathUtils.lerp(0.28, -0.28, upEase);
          torsoPitch = THREE.MathUtils.lerp(0.28, -0.36, upEase);
          lHipX = THREE.MathUtils.lerp(-0.58, -0.84, upEase);
          lKneeX = THREE.MathUtils.lerp(1.0, 1.22, upEase);
          rShoulderX = THREE.MathUtils.lerp(0.84, -2.66, upEase);
          lShoulderX = THREE.MathUtils.lerp(0.84, -2.62, upEase);
          rElbowX = -0.1;
        }
      } else {
        const dir = f.comboStep === 2 ? -1 : 1;
        const a = THREE.MathUtils.clamp((p - 0.22) / 0.28, 0, 1);
        const aEase = 1 - Math.pow(1 - a, 4);
        const sweep =
          p < 0.22
            ? -1.42 * dir * THREE.MathUtils.smoothstep(p, 0, 0.22)
            : p < 0.72
              ? THREE.MathUtils.lerp(-1.42 * dir, 1.38 * dir, aEase)
              : THREE.MathUtils.lerp(1.38 * dir, 0, THREE.MathUtils.smoothstep(p, 0.72, 1.0));
        bodyRoll = sweep * 0.085;
        bodyPitch = p < 0.22 ? 0.08 : 0.25 * aEase;
        bodyBob = p < 0.22 ? -0.08 : -0.05 * aEase;
        lHipX = p < 0.22 ? -0.34 : -0.62 * aEase;
        lKneeX = p < 0.22 ? 0.56 : 0.86 * aEase;
        rHipX = p < 0.22 ? 0.28 : 0.5 * aEase;
        torsoYaw = sweep * 0.74;
        headYaw = -torsoYaw * 0.68;
        rShoulderX = -1.24;
        rShoulderZ = -0.36 + sweep * 0.54;
        rElbowX = -0.22;
        lShoulderX = f.weapon.kind === 'kodachi' ? -1.24 : -0.65;
        lShoulderZ = f.weapon.kind === 'kodachi' ? 0.36 - sweep * 0.54 : 0.35;
      }
    }

    if (f.isGrounded && f.jumpSquashTimer > 0 && f.attackPhase === 'none' && !f.isDashing) {
      const sq = Math.sin((f.jumpSquashTimer / 0.2) * Math.PI);
      bodyBob -= sq * 0.16;
      bodyPitch += sq * 0.1;
      lKneeX += sq * 0.52;
      rKneeX += sq * 0.52;
      lHipX -= sq * 0.25;
      rHipX -= sq * 0.25;
    }

    if (rig.isCustomModel) {
      if (rig.attackSlashMesh) {
        if (f.attackPhase === 'active' && !f.weapon.isRanged) {
          const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
          if (f.isSpecialAttack && f.weapon.kind === 'kodachi') {
            rig.attackSlashMesh.visible = true;
            if (p < 0.66) {
              const subP = ((p - 0.14) / 0.18) % 1;
              const altDir = Math.floor((p - 0.14) / 0.18) % 2 === 0 ? 1 : -1;
              rig.attackSlashMesh.rotation.z = altDir * 0.18;
              rig.attackSlashMesh.rotation.y = Math.PI / 2 + (-1.1 + subP * 2.2) * altDir;
              rig.attackSlashMesh.scale.setScalar(1.25 + subP * 0.35);
            } else {
              const upP = THREE.MathUtils.clamp((p - 0.72) / 0.2, 0, 1);
              rig.attackSlashMesh.rotation.z = Math.PI * 0.5;
              rig.attackSlashMesh.rotation.y = Math.PI / 2 - 0.8 + upP * 1.6;
              rig.attackSlashMesh.scale.setScalar(1.45 + upP * 0.5);
            }
            (rig.attackSlashMesh.material as THREE.MeshBasicMaterial).opacity = 0.92;
          } else if (f.isSpecialAttack && f.weapon.kind === 'hammer') {
            const slamProg = THREE.MathUtils.clamp((p - 0.54) / 0.26, 0, 1);
            rig.attackSlashMesh.visible = true;
            rig.attackSlashMesh.rotation.z = Math.PI * 0.5;
            rig.attackSlashMesh.rotation.y = Math.PI / 2 - 0.65 + slamProg * 1.3;
            rig.attackSlashMesh.scale.setScalar(1.42 + slamProg * 0.65);
            (rig.attackSlashMesh.material as THREE.MeshBasicMaterial).opacity =
              (1 - slamProg) * 0.94;
          } else {
            const swingProg = THREE.MathUtils.clamp((p - 0.28) / 0.37, 0, 1);
            rig.attackSlashMesh.visible = true;
            rig.attackSlashMesh.rotation.z = f.airAttacking
              ? 1.15
              : f.comboStep === 2
                ? 0.35
                : f.comboStep === 3 || f.isSpecialAttack
                  ? 1.45
                  : -0.25;
            rig.attackSlashMesh.rotation.y =
              Math.PI / 2 + (f.comboStep === 2 ? -0.9 + swingProg * 1.8 : -0.4 + swingProg * 0.8);
            rig.attackSlashMesh.scale.setScalar((f.isSpecialAttack ? 1.25 : 0.85) + swingProg * 0.45);
            (rig.attackSlashMesh.material as THREE.MeshBasicMaterial).opacity =
              (1 - swingProg) * 0.9;
          }
        } else {
          rig.attackSlashMesh.visible = false;
        }
      }

      if (rig.specialAnimActive) {
        return;
      }

      if (!rig.mixer) {
        const holdingRifle = Boolean(
          f.weapon.isRanged &&
            (rig.hasHangarWeapon || rig.customCharacterId === 'hunter' || rig.hasBuiltInWeapon),
        );
        const holdingLance = f.weapon.kind === 'lance';
        const isBackward = forwardDot < -0.4;
        let customTarget: CustomBonePose = createZeroCustomPose();

        if (f.stunTimer > 0) {
          customTarget.bodyPitch = bodyPitch;
          customTarget.bodyBob = bodyBob;
          customTarget.lHipX = lHipX;
          customTarget.rHipX = rHipX;
          customTarget.lKneeX = lKneeX;
          customTarget.rKneeX = rKneeX;
          customTarget.torsoPitch = -0.25;
          customTarget.lShoulderX = -0.35;
          customTarget.rShoulderX = -0.35;
          customTarget.lShoulderZ = 0.45;
          customTarget.rShoulderZ = -0.45;
        } else if (holdingRifle) {
          if (f.isDashing) {
            customTarget = buildCustomRiflePose({
              mode: 'dash',
              elapsed: t,
              dashProgress,
              dashForward,
              dashRight,
              isPartRig: rig.isPartArticulated,
            });
          } else if (!f.isGrounded) {
            if (f.attackPhase !== 'none') {
              const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
              customTarget = buildCustomRiflePose({
                mode: 'air_fire',
                elapsed: t,
                fireProgress: p,
                shotVariant: f.isSpecialAttack ? 3 : 1,
                velY: f.velY,
                strafeRatio,
                isPartRig: rig.isPartArticulated,
                isAiming: f.isAiming,
              });
            } else {
              customTarget = buildCustomRiflePose({
                mode: 'air_aim',
                elapsed: t,
                velY: f.velY,
                strafeRatio,
                isPartRig: rig.isPartArticulated,
                isAiming: f.isAiming,
              });
            }
          } else if (f.isGuarding) {
            customTarget.bodyPitch = bodyPitch;
            customTarget.bodyBob = bodyBob;
            customTarget.lHipX = lHipX;
            customTarget.rHipX = rHipX;
            customTarget.lKneeX = lKneeX;
            customTarget.rKneeX = rKneeX;
            customTarget.torsoYaw = 0.35;
            customTarget.lShoulderX = -1.1;
            customTarget.lShoulderZ = -0.35;
            customTarget.lElbowX = -0.65;
            customTarget.rShoulderX = -0.4;
            customTarget.rElbowX = -0.5;
          } else if (f.attackPhase !== 'none') {
            const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
            customTarget = buildCustomRiflePose({
              mode: 'fire',
              elapsed: t,
              fireProgress: p,
              shotVariant: f.isSpecialAttack ? 3 : 1,
              isPartRig: rig.isPartArticulated,
              isAiming: f.isAiming,
            });
          } else if (speedRatio > 0.1) {
            customTarget = buildCustomRiflePose({
              mode: 'low_ready',
              elapsed: t,
              speedRatio,
              stridePhase: f.stridePhase,
              backward: isBackward,
              strafeRatio,
              isPartRig: rig.isPartArticulated,
              isAiming: f.isAiming,
            });
          } else {
            customTarget = buildCustomRiflePose({
              mode: 'aim',
              elapsed: t,
              isPartRig: rig.isPartArticulated,
              isAiming: f.isAiming,
            });
          }
        } else if (holdingLance) {
          if (f.isDashing) {
            customTarget = buildCustomLancePose({
              mode: 'dash',
              elapsed: t,
              dashProgress,
              dashForward,
              dashRight,
            });
          } else if (!f.isGrounded && !f.isSpecialAttack) {
            if (f.attackPhase !== 'none') {
              const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
              customTarget = buildCustomLancePose({
                mode: 'air_strike',
                elapsed: t,
                strikeProgress: p,
                velY: f.velY,
                strafeRatio,
              });
            } else {
              customTarget = buildCustomLancePose({
                mode: 'jump',
                elapsed: t,
                velY: f.velY,
                strafeRatio,
              });
            }
          } else if (f.isGuarding) {
            customTarget = buildCustomLancePose({ mode: 'guard', elapsed: t });
          } else if (f.attackPhase !== 'none') {
            const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
            customTarget = buildCustomLancePose({
              mode: 'strike',
              elapsed: t,
              strikeProgress: p,
              strikeVariant: f.isSpecialAttack ? 3 : f.comboStep,
              isSpecial: f.isSpecialAttack,
            });
          } else if (speedRatio > 0.08) {
            customTarget = buildCustomLancePose({
              mode: 'run',
              elapsed: t,
              speedRatio,
              stridePhase: f.stridePhase,
              backward: isBackward,
              strafeRatio,
            });
          } else {
            customTarget = buildCustomLancePose({ mode: 'ready', elapsed: t });
          }
        } else {
          const isHeavy = f.weapon.kind === 'hammer';
          const isTwinBlade = f.weapon.kind === 'kodachi';
          if (f.isDashing) {
            customTarget = buildCustomMeleePose({
              mode: 'dash',
              elapsed: t,
              dashProgress,
              dashForward,
              dashRight,
              isHeavy,
              isTwinBlade,
            });
          } else if (!f.isGrounded && !(f.isSpecialAttack && (isTwinBlade || isHeavy))) {
            const p =
              f.attackPhase !== 'none'
                ? 1 - Math.max(0, f.attackTimer / f.attackDuration)
                : 0;
            customTarget = buildCustomJumpPose({
              mode: f.attackPhase !== 'none' ? 'air_strike' : 'jump',
              velY: f.velY,
              strikeProgress: p,
              isHeavy,
              isTwinBlade,
              elapsed: t,
              strafeRatio,
            });
          } else if (f.isGuarding) {
            customTarget = buildCustomMeleePose({
              mode: 'guard',
              elapsed: t,
              isHeavy,
              isTwinBlade,
            });
          } else if (f.attackPhase !== 'none') {
            const p = 1 - Math.max(0, f.attackTimer / f.attackDuration);
            customTarget = buildCustomMeleePose({
              mode: 'strike',
              elapsed: t,
              strikeProgress: p,
              strikeVariant: f.isSpecialAttack ? 3 : f.comboStep,
              isHeavy,
              isTwinBlade,
              isWindSpecial: f.isSpecialAttack && isTwinBlade,
              isHammerSpecial: f.isSpecialAttack && isHeavy,
            });
          } else if (speedRatio > 0.08) {
            customTarget = buildCustomMeleePose({
              mode: 'run',
              elapsed: t,
              speedRatio,
              stridePhase: f.stridePhase,
              backward: isBackward,
              strafeRatio,
              isHeavy,
              isTwinBlade,
            });
          } else {
            customTarget = buildCustomMeleePose({
              mode: 'ready',
              elapsed: t,
              isHeavy,
              isTwinBlade,
            });
          }
        }

        if (f.isGrounded && f.jumpSquashTimer > 0 && f.attackPhase === 'none' && !f.isDashing) {
          const sq = Math.sin((f.jumpSquashTimer / 0.2) * Math.PI);
          customTarget.bodyBob -= sq * 0.14;
          customTarget.bodyPitch += sq * 0.08;
          customTarget.lKneeX += sq * 0.48;
          customTarget.rKneeX += sq * 0.48;
          customTarget.lHipX -= sq * 0.22;
          customTarget.rHipX -= sq * 0.22;
        }

        const freeAimActive =
          f.isPlayer && !this.lockOnActive && f.stunTimer <= 0 && !f.isDashing;
        if (freeAimActive) {
          const meleeStrikeScale =
            !f.weapon.isRanged && f.attackPhase !== 'none'
              ? f.isGrounded
                ? 0.18
                : 0.42
              : 1;
          customTarget.torsoYaw += f.aimTorsoYaw * 0.82 * meleeStrikeScale;
          customTarget.torsoPitch += f.aimTorsoPitch * 0.78 * meleeStrikeScale;
          customTarget.headYaw += f.aimTorsoYaw * 0.28 * meleeStrikeScale;
          customTarget.headPitch += f.aimTorsoPitch * 0.48 * meleeStrikeScale;
          if (holdingRifle) {
            customTarget.rShoulderX += f.aimTorsoPitch * 0.45;
            customTarget.lShoulderX += f.aimTorsoPitch * 0.45;
          }
        }

        applyCustomBonePose(
          rig,
          customTarget,
          dt,
          f.attackPhase !== 'none' || f.isGuarding || f.isDashing ? 22 : 16,
          freeAimActive && f.attackPhase === 'none',
        );
      }
      return;
    }

    const freeAimModular =
      f.isPlayer && !this.lockOnActive && f.stunTimer <= 0 && !f.isDashing;
    let headPitch = 0;
    if (freeAimModular) {
      torsoYaw += f.aimTorsoYaw * 0.8;
      torsoPitch += f.aimTorsoPitch * 0.75;
      headYaw += f.aimTorsoYaw * 0.32;
      headPitch = f.aimTorsoPitch * 0.48;
      if (f.weapon.isRanged) {
        rShoulderX += f.aimTorsoPitch * 0.45;
        lShoulderX += f.aimTorsoPitch * 0.45;
      }
    }

    const jLambda = f.attackPhase !== 'none' || f.isDashing ? 22 : 16;
    rig.bodyGroup.rotation.order = 'YXZ';
    rig.bodyGroup.rotation.x = THREE.MathUtils.damp(rig.bodyGroup.rotation.x, bodyPitch, jLambda, dt);
    rig.bodyGroup.rotation.z = THREE.MathUtils.damp(rig.bodyGroup.rotation.z, bodyRoll, jLambda, dt);
    rig.bodyGroup.position.y = THREE.MathUtils.damp(rig.bodyGroup.position.y, bodyBob, jLambda, dt);
    rig.torsoPivot.rotation.y = THREE.MathUtils.damp(rig.torsoPivot.rotation.y, torsoYaw, jLambda, dt);
    rig.torsoPivot.rotation.x = THREE.MathUtils.damp(rig.torsoPivot.rotation.x, torsoPitch, jLambda, dt);
    rig.headPivot.rotation.y = THREE.MathUtils.damp(rig.headPivot.rotation.y, headYaw, jLambda, dt);
    rig.headPivot.rotation.x = THREE.MathUtils.damp(rig.headPivot.rotation.x, headPitch, jLambda, dt);

    rig.leftHip.rotation.x = THREE.MathUtils.damp(rig.leftHip.rotation.x, lHipX, jLambda, dt);
    rig.leftHip.rotation.z = THREE.MathUtils.damp(rig.leftHip.rotation.z, lHipZ, jLambda, dt);
    rig.rightHip.rotation.x = THREE.MathUtils.damp(rig.rightHip.rotation.x, rHipX, jLambda, dt);
    rig.rightHip.rotation.z = THREE.MathUtils.damp(rig.rightHip.rotation.z, rHipZ, jLambda, dt);
    rig.leftKnee.rotation.x = THREE.MathUtils.damp(rig.leftKnee.rotation.x, lKneeX, jLambda, dt);
    rig.rightKnee.rotation.x = THREE.MathUtils.damp(rig.rightKnee.rotation.x, rKneeX, jLambda, dt);

    rig.leftShoulder.rotation.x = THREE.MathUtils.damp(rig.leftShoulder.rotation.x, lShoulderX, jLambda, dt);
    rig.leftShoulder.rotation.y = THREE.MathUtils.damp(rig.leftShoulder.rotation.y, lShoulderY, jLambda, dt);
    rig.leftShoulder.rotation.z = THREE.MathUtils.damp(rig.leftShoulder.rotation.z, lShoulderZ, jLambda, dt);
    rig.leftElbow.rotation.x = THREE.MathUtils.damp(rig.leftElbow.rotation.x, lElbowX, jLambda, dt);

    rig.rightShoulder.rotation.x = THREE.MathUtils.damp(rig.rightShoulder.rotation.x, rShoulderX, jLambda, dt);
    rig.rightShoulder.rotation.z = THREE.MathUtils.damp(rig.rightShoulder.rotation.z, rShoulderZ, jLambda, dt);
    rig.rightElbow.rotation.x = THREE.MathUtils.damp(rig.rightElbow.rotation.x, rElbowX, jLambda, dt);

    const flameTarget = f.isDashing ? 1.8 : !f.isGrounded ? 1.3 : speedRatio > 0.25 ? 0.65 : 0.18;
    rig.thrusterFlames.forEach((flame) => {
      flame.scale.y = THREE.MathUtils.damp(flame.scale.y, flameTarget, 14, dt);
    });
  }

  private updateProjectiles(dt: number) {
    for (let i = 0; i < this.projectiles.length; i++) {
      const p = this.projectiles[i];
      if (!p.active) continue;

      p.prevPos.copy(p.pos);
      p.life -= dt;
      p.pos.addScaledVector(p.vel, dt);
      p.mesh.position.copy(p.pos);

      if (p.life <= 0 || p.pos.y < 0.05 || Math.hypot(p.pos.x, p.pos.z) > this.arena.radius + 0.5) {
        p.active = false;
        p.mesh.visible = false;
        continue;
      }

      const target = p.ownerIsPlayer ? this.enemy : this.player;
      const attacker = p.ownerIsPlayer ? this.player : this.enemy;
      let hitDone = false;

      // Sub-step along prevPos -> pos so high-velocity sniper bolts never tunnel
      for (let s = 1; s <= 3; s++) {
        const alpha = s / 3;
        const sx = THREE.MathUtils.lerp(p.prevPos.x, p.pos.x, alpha);
        const sy = THREE.MathUtils.lerp(p.prevPos.y, p.pos.y, alpha);
        const sz = THREE.MathUtils.lerp(p.prevPos.z, p.pos.z, alpha);

        let hitObs: ObstacleState | null = null;
        for (let j = 0; j < this.obstacleStates.length; j++) {
          const o = this.obstacleStates[j];
          if (o.destroyed) continue;
          if (
            sy <= o.data.height + 0.15 &&
            Math.hypot(sx - o.data.x, sz - o.data.z) < o.data.radius + 0.15
          ) {
            hitObs = o;
            break;
          }
        }
        if (hitObs) {
          if (hitObs.data.destructible) {
            this.damageDestructibleObstacle(hitObs, p.damage, attacker);
          } else {
            this.spawnSparks(sx, sy, sz, 6, '#38BDF8');
          }
          p.active = false;
          p.mesh.visible = false;
          hitDone = true;
          break;
        }

        const dx = sx - target.pos.x;
        const dz = sz - target.pos.z;
        if (
          Math.hypot(dx, dz) < 0.88 &&
          sy >= target.pos.y - 0.15 &&
          sy <= target.pos.y + 2.15
        ) {
          this.applyHit(attacker, target, p.damage, p.isSpecial ? 6.5 : 2.6, p.isSpecial);
          p.active = false;
          p.mesh.visible = false;
          hitDone = true;
          break;
        }
      }

      if (hitDone) continue;
    }

    // Update Homing Stinger Missiles
    for (let i = 0; i < this.missiles.length; i++) {
      const m = this.missiles[i];
      if (!m.active) continue;

      m.age += dt;
      m.life -= dt;
      const target = m.ownerIsPlayer ? this.enemy : this.player;
      const attacker = m.ownerIsPlayer ? this.player : this.enemy;

      if (m.age > 0.13) {
        _v1.set(target.pos.x, target.pos.y + 1.02, target.pos.z).sub(m.pos).normalize();
        const steerRate = m.age < 0.35 ? 6.5 : 11.5;
        const curDir = _v2.copy(m.vel).normalize();
        curDir.lerp(_v1, Math.min(1, steerRate * dt)).normalize();
        m.vel.copy(curDir).multiplyScalar(27.5);
      } else {
        m.vel.y -= 4.5 * dt;
      }

      m.pos.addScaledVector(m.vel, dt);
      m.mesh.position.copy(m.pos);
      if (m.vel.lengthSq() > 0.01) {
        m.mesh.quaternion.setFromUnitVectors(_zAxis, _v2.copy(m.vel).normalize());
      }

      m.smokeTimer -= dt;
      if (m.smokeTimer <= 0) {
        m.smokeTimer = 0.045;
        this.spawnSparks(m.pos.x, m.pos.y, m.pos.z, 1, '#FDBA74');
      }

      let detonate = m.life <= 0 || m.pos.y <= 0.15;
      let hitTarget = false;

      if (!detonate) {
        const tDist = Math.hypot(m.pos.x - target.pos.x, m.pos.z - target.pos.z);
        if (
          tDist < 1.05 &&
          m.pos.y >= target.pos.y - 0.1 &&
          m.pos.y <= target.pos.y + 2.25
        ) {
          detonate = true;
          hitTarget = true;
        }
      }

      if (!detonate) {
        for (let j = 0; j < this.obstacleStates.length; j++) {
          const o = this.obstacleStates[j];
          if (o.destroyed) continue;
          if (
            m.pos.y <= o.data.height + 0.2 &&
            Math.hypot(m.pos.x - o.data.x, m.pos.z - o.data.z) < o.data.radius + 0.25
          ) {
            if (o.data.destructible) {
              this.damageDestructibleObstacle(o, m.damage, attacker);
            }
            detonate = true;
            break;
          }
        }
      }

      if (detonate) {
        if (hitTarget) {
          this.applyHit(attacker, target, m.damage, m.isFinal ? 7.5 : 3.4, m.isFinal);
        } else if (m.pos.distanceTo(target.pos) < 2.2) {
          this.applyHit(attacker, target, m.damage * 0.6, 3.2, false);
        }
        this.spawnShockwave(m.pos.x, Math.max(0.15, m.pos.y), m.pos.z, 2.2, '#F97316');
        this.spawnSparks(m.pos.x, m.pos.y, m.pos.z, 10, '#FDE047');
        soundEngine.playHit(m.isFinal);

        if (m.customVisual) {
          m.mesh.remove(m.customVisual);
          m.customVisual = null;
        }
        m.active = false;
        m.mesh.visible = false;
      }
    }
  }

  private updateVfx(dt: number) {
    for (let i = 0; i < this.sparks.length; i++) {
      const s = this.sparks[i];
      if (!s.active) continue;
      s.life -= dt;
      s.vel.y -= 12 * dt;
      s.pos.addScaledVector(s.vel, dt);
      if (s.pos.y < 0.05) {
        s.pos.y = 0.05;
        s.vel.y *= -0.4;
      }
      s.mesh.position.copy(s.pos);
      const scale = Math.max(0.01, s.life / s.maxLife);
      s.mesh.scale.setScalar(scale);
      if (s.life <= 0) {
        s.active = false;
        s.mesh.visible = false;
      }
    }

    for (let i = 0; i < this.shockwaves.length; i++) {
      const r = this.shockwaves[i];
      if (!r.active) continue;
      r.life -= dt;
      const prog = 1 - Math.max(0, r.life / r.maxLife);
      const curRadius = 0.25 + prog * r.maxRadius;
      r.mesh.scale.setScalar(curRadius);
      (r.mesh.material as THREE.MeshBasicMaterial).opacity = (1 - prog) * 0.85;
      if (r.life <= 0) {
        r.active = false;
        r.mesh.visible = false;
      }
    }

    for (let i = 0; i < this.lightningBolts.length; i++) {
      const b = this.lightningBolts[i];
      if (!b.active) continue;
      b.life -= dt;
      const alpha = Math.max(0, b.life / b.maxLife);
      const flicker = Math.random() > 0.2 ? 1 : 0.45;
      for (let m = 0; m < b.mats.length; m++) {
        b.mats[m].opacity = alpha * flicker;
      }
      if (b.life <= 0) {
        b.active = false;
        b.group.visible = false;
      }
    }

    if (this.thunderShotActive && this.thunderShotCaster && this.thunderShotTarget) {
      this.thunderShotTimer += dt;
      const alpha = Math.min(1, this.thunderShotTimer / Math.max(0.08, this.thunderShotDuration));
      const targetChestX = this.thunderShotTarget.pos.x;
      const targetChestY = this.thunderShotTarget.pos.y + 1.08;
      const targetChestZ = this.thunderShotTarget.pos.z;

      this.thunderShotPos.set(
        THREE.MathUtils.lerp(this.thunderShotOrigin.x, targetChestX, alpha),
        THREE.MathUtils.lerp(this.thunderShotOrigin.y, targetChestY, alpha),
        THREE.MathUtils.lerp(this.thunderShotOrigin.z, targetChestZ, alpha),
      );

      this.thunderSpearOrb.visible = true;
      this.thunderSpearOrb.position.copy(this.thunderShotPos);
      this.thunderSpearOrb.scale.setScalar(1.55 + Math.sin(this.elapsedTime * 54) * 0.25);
      this.thunderSpearArcA.rotation.x += dt * 18;
      this.thunderSpearArcA.rotation.y += dt * 14;
      this.thunderSpearArcB.rotation.z -= dt * 20;
      this.thunderLight.position.copy(this.thunderShotPos);
      this.thunderFlashIntensity = Math.max(this.thunderFlashIntensity, 22);

      this.thunderShotArcCooldown -= dt;
      if (this.thunderShotArcCooldown <= 0) {
        this.thunderShotArcCooldown = 0.04;
        this.thunderShotCaster.rig.bladeTipPoint.getWorldPosition(_v1);
        if (_v1.y < this.thunderShotCaster.pos.y + 0.25) {
          _v1.copy(this.thunderShotOrigin);
        }
        this.spawnLightningBolt(
          _v1.x,
          _v1.y,
          _v1.z,
          this.thunderShotPos.x,
          this.thunderShotPos.y,
          this.thunderShotPos.z,
          1.35,
          '#38BDF8',
        );
        this.spawnLightningBolt(
          this.thunderShotOrigin.x,
          this.thunderShotOrigin.y,
          this.thunderShotOrigin.z,
          this.thunderShotPos.x,
          this.thunderShotPos.y,
          this.thunderShotPos.z,
          0.95,
          '#E0F2FE',
        );
        this.spawnSparks(this.thunderShotPos.x, this.thunderShotPos.y, this.thunderShotPos.z, 4, '#60A5FA');
      }

      if (alpha >= 1) {
        const caster = this.thunderShotCaster;
        const target = this.thunderShotTarget;
        this.thunderShotActive = false;
        this.thunderShotCaster = null;
        this.thunderShotTarget = null;
        if (!caster.hasHitCurrentSwing) {
          caster.hasHitCurrentSwing = true;
          this.executeAttackImpact(caster, target);
        }
      }
    }

    if (this.thunderAuraTimer > 0) {
      this.thunderAuraTimer = Math.max(0, this.thunderAuraTimer - dt);
      const activeLanceFighter =
        this.player.isSpecialAttack && this.player.weapon.kind === 'lance'
          ? this.player
          : this.enemy.isSpecialAttack && this.enemy.weapon.kind === 'lance'
            ? this.enemy
            : null;
      if (activeLanceFighter) {
        this.thunderAuraGroup.visible = true;
        this.thunderAuraGroup.position.copy(activeLanceFighter.pos);
        this.thunderOuterRing.rotation.y += dt * 4.8;
        this.thunderInnerRing.rotation.y -= dt * 7.2;
        const pulse = 1 + Math.sin(this.elapsedTime * 28) * 0.14;
        this.thunderOuterRing.scale.setScalar(pulse);
        this.thunderPillar.scale.set(
          0.85 + Math.sin(this.elapsedTime * 34) * 0.25,
          1,
          0.85 + Math.cos(this.elapsedTime * 34) * 0.25,
        );
        const lanceProg =
          1 - activeLanceFighter.attackTimer / Math.max(0.1, activeLanceFighter.attackDuration);
        if (!this.thunderShotActive) {
          const chargeScale = THREE.MathUtils.lerp(0.48, 1.38, Math.min(1, lanceProg / 0.77));
          this.thunderSpearOrb.scale.setScalar(chargeScale * (0.94 + Math.sin(this.elapsedTime * 44) * 0.16));
          this.thunderSpearArcA.rotation.x += dt * 11;
          this.thunderSpearArcA.rotation.z += dt * 8;
          this.thunderSpearArcB.rotation.y -= dt * 13;
        }

        for (let i = 0; i < this.thunderCurtainStreaks.length; i++) {
          const st = this.thunderCurtainStreaks[i];
          const ud = st.userData as { baseY: number; speed: number; phase: number };
          const cyc = ((this.elapsedTime * ud.speed + ud.phase) % 4.5) / 4.5;
          st.position.y = ud.baseY + cyc * 3.8;
          (st.material as THREE.MeshBasicMaterial).opacity = Math.sin(cyc * Math.PI) * 0.62;
        }

        const showCone =
          (this.thunderCutInProgress >= 0.86 || this.thunderShotActive || activeLanceFighter.attackPhase === 'active') &&
          lanceProg < 0.94;
        if (showCone) {
          this.thunderLanceCone.visible = true;
          activeLanceFighter.rig.bladeTipPoint.getWorldPosition(_v1);
          if (_v1.y < activeLanceFighter.pos.y + 0.25) {
            _v1.set(
              activeLanceFighter.pos.x + Math.sin(activeLanceFighter.yaw) * 0.9,
              activeLanceFighter.pos.y + 1.25,
              activeLanceFighter.pos.z + Math.cos(activeLanceFighter.yaw) * 0.9,
            );
          }
          this.thunderLanceCone.position.copy(_v1);
          this.thunderLanceCone.rotation.set(0, activeLanceFighter.yaw, 0);
          const conePulse = 1 + Math.sin(this.elapsedTime * 48) * 0.14;
          this.thunderLanceCone.scale.set(conePulse, conePulse, 1 + Math.cos(this.elapsedTime * 36) * 0.18);
          for (let r = 0; r < this.thunderConeRings.length; r++) {
            this.thunderConeRings[r].rotation.z += dt * (9 + r * 4);
          }
        } else {
          this.thunderLanceCone.visible = false;
        }
      } else {
        this.thunderAuraTimer = 0;
        this.thunderAuraGroup.visible = false;
        if (!this.thunderShotActive) {
          this.thunderSpearOrb.visible = false;
        }
        this.thunderLanceCone.visible = false;
      }
    } else {
      this.thunderAuraGroup.visible = false;
      if (!this.thunderShotActive) {
        this.thunderSpearOrb.visible = false;
      }
      this.thunderLanceCone.visible = false;
    }

    if (this.thunderFlashIntensity > 0.05) {
      this.thunderFlashIntensity = THREE.MathUtils.damp(this.thunderFlashIntensity, 0, 12, dt);
      this.thunderLight.intensity = this.thunderFlashIntensity;
    } else {
      this.thunderFlashIntensity = 0;
      this.thunderLight.intensity = 0;
    }
  }

  private updateCamera(dt: number) {
    const pPos = this.player.pos;
    const ePos = this.enemy.pos;
    const isSniperAds = this.player.isAiming;
    const cutIn = this.thunderCutInProgress;
    const isThunderLunge =
      cutIn < 0 &&
      this.player.isSpecialAttack &&
      this.player.weapon.kind === 'lance' &&
      this.player.attackPhase === 'active';

    let thunderShotIdx = -1;
    if (cutIn >= 0) {
      const fx = Math.sin(this.player.yaw);
      const fz = Math.cos(this.player.yaw);
      // Character's right-arm (spear) side = (-fz, 0, fx); left-arm (shield) side = (fz, 0, -fx)
      const rx = -fz;
      const rz = fx;
      const lx = fz;
      const lz = -fx;

      if (cutIn < 0.33) {
        // Shot 0 (00:00): Medium front-right upward aura charge close-up
        thunderShotIdx = 0;
        const s = cutIn / 0.33;
        const fwd = THREE.MathUtils.lerp(2.1, 1.68, s);
        const rightSide = THREE.MathUtils.lerp(0.62, 0.26, s);
        const up = THREE.MathUtils.lerp(1.02, 1.24, s);
        _desiredCamPos.set(
          pPos.x + fx * fwd + rx * rightSide,
          pPos.y + up,
          pPos.z + fz * fwd + rz * rightSide,
        );
        _desiredLook.set(
          pPos.x + fx * 0.15 + rx * 0.05,
          pPos.y + 1.28,
          pPos.z + fz * 0.15 + rz * 0.05,
        );
      } else if (cutIn < 0.6) {
        // Shot 1 (00:01): Extreme shield-rim close-up looking up at Achilles' helmet & visor
        thunderShotIdx = 1;
        const s = (cutIn - 0.33) / 0.27;
        const fwd = THREE.MathUtils.lerp(1.48, 1.32, s);
        const leftSide = THREE.MathUtils.lerp(0.76, 0.58, s);
        const up = THREE.MathUtils.lerp(1.16, 1.3, s);
        _desiredCamPos.set(
          pPos.x + fx * fwd + lx * leftSide,
          pPos.y + up,
          pPos.z + fz * fwd + lz * leftSide,
        );
        _desiredLook.set(
          pPos.x + fx * 0.12 + lx * 0.06,
          pPos.y + 1.4,
          pPos.z + fz * 0.12 + lz * 0.06,
        );
      } else if (cutIn < 0.9) {
        // Shot 2 (00:02): Heroic 3/4 shield-side profile showing drawn-back blazing Lightning Lance
        thunderShotIdx = 2;
        const s = (cutIn - 0.6) / 0.3;
        const ease = 1 - Math.pow(1 - s, 2);
        const fwd = THREE.MathUtils.lerp(1.88, 1.56, ease);
        const leftSide = THREE.MathUtils.lerp(1.32, 1.66, ease);
        const up = THREE.MathUtils.lerp(1.18, 1.34, ease);
        _desiredCamPos.set(
          pPos.x + fx * fwd + lx * leftSide,
          pPos.y + up,
          pPos.z + fz * fwd + lz * leftSide,
        );
        _desiredLook.set(
          pPos.x + fx * 0.18 + rx * 0.14,
          pPos.y + 1.28,
          pPos.z + fz * 0.18 + rz * 0.14,
        );
      } else {
        // Shot 3 (00:03): Forward vortex burst view looking down the surging cyan energy cone
        thunderShotIdx = 3;
        const s = (cutIn - 0.9) / 0.1;
        const fwd = THREE.MathUtils.lerp(2.5, 3.1, s);
        _desiredCamPos.set(
          pPos.x + fx * fwd + rx * 0.34,
          pPos.y + 1.12,
          pPos.z + fz * fwd + rz * 0.34,
        );
        _desiredLook.set(
          pPos.x + fx * 0.45,
          pPos.y + 1.16,
          pPos.z + fz * 0.45,
        );
      }
    } else if (
      this.lockOnActive ||
      (this.player.isSpecialAttack && this.player.weapon.kind === 'lance')
    ) {
      _v1.subVectors(ePos, pPos);
      _v1.y = 0;
      const dist = Math.max(0.5, _v1.length());
      _v1.divideScalar(dist);

      const camBack = isSniperAds
        ? 2.85
        : THREE.MathUtils.clamp(5.8 + dist * 0.22, 6.0, 9.2);
      const camHeight = isSniperAds
        ? 1.72
        : THREE.MathUtils.clamp(3.2 + dist * 0.14, 3.4, 5.4);
      const sideOffset = isSniperAds ? 0.72 : 1.15;

      _v2.set(-_v1.z, 0, _v1.x);

      _desiredCamPos.set(
        pPos.x - _v1.x * camBack + _v2.x * sideOffset,
        camHeight + pPos.y * 0.9,
        pPos.z - _v1.z * camBack + _v2.z * sideOffset,
      );

      _desiredLook.set(
        THREE.MathUtils.lerp(pPos.x, ePos.x, isSniperAds ? 0.88 : 0.55),
        1.15 + THREE.MathUtils.lerp(pPos.y, ePos.y, isSniperAds ? 0.88 : 0.55) * 0.88,
        THREE.MathUtils.lerp(pPos.z, ePos.z, isSniperAds ? 0.88 : 0.55),
      );
    } else {
      if (Math.abs(this.pointerEdgeRate) > 0.01) {
        this.camOrbitYaw += this.pointerEdgeRate * this.mouseSensitivity * dt;
      }
      if (this.keys.has('KeyQ')) {
        this.camOrbitYaw += 2.2 * dt;
      }
      if (this.keys.has('KeyC')) {
        this.camOrbitYaw -= 2.2 * dt;
      }

      _camForward.set(-Math.sin(this.camOrbitYaw), 0, -Math.cos(this.camOrbitYaw));
      _camRight.set(-_camForward.z, 0, _camForward.x);

      if (isSniperAds) {
        const adsDist = 2.3;
        const adsPitch = this.camOrbitPitch - 0.28;
        const cosAds = Math.cos(adsPitch);
        const sinAds = Math.sin(adsPitch);
        _desiredCamPos.set(
          pPos.x - _camForward.x * adsDist + _camRight.x * 0.58,
          pPos.y + 1.54 + sinAds * 1.15,
          pPos.z - _camForward.z * adsDist + _camRight.z * 0.58,
        );
        _desiredLook.set(
          _desiredCamPos.x + _camForward.x * cosAds * 28,
          Math.max(0.25, _desiredCamPos.y - sinAds * 28),
          _desiredCamPos.z + _camForward.z * cosAds * 28,
        );
      } else {
        const dist = this.camDistance;
        const cosP = Math.cos(this.camOrbitPitch);
        const camBaseY = this.player.groundY + (pPos.y - this.player.groundY) * 0.82;

        _desiredCamPos.set(
          pPos.x + Math.sin(this.camOrbitYaw) * cosP * dist + _camRight.x * 0.42,
          camBaseY + 1.28 + Math.sin(this.camOrbitPitch) * dist,
          pPos.z + Math.cos(this.camOrbitYaw) * cosP * dist + _camRight.z * 0.42,
        );

        const lookY = Math.max(
          0.4,
          camBaseY + 1.26 - Math.sin(this.camOrbitPitch) * 2.1,
        );
        _desiredLook.set(
          pPos.x + _camForward.x * 3.4 + _camRight.x * 0.22,
          lookY,
          pPos.z + _camForward.z * 3.4 + _camRight.z * 0.22,
        );
      }
    }

    _desiredCamPos.y = Math.max(0.75, _desiredCamPos.y);

    const snapCut = thunderShotIdx !== this.lastThunderShot;
    this.lastThunderShot = thunderShotIdx;

    if (snapCut) {
      this.camera.position.copy(_desiredCamPos);
      _currentCamLook.copy(_desiredLook);
    } else {
      const posLambda = cutIn >= 0 ? 14 : isSniperAds ? 18 : this.lockOnActive ? 7.5 : 16.5;
      const lookLambda = cutIn >= 0 ? 20 : isSniperAds ? 24 : this.lockOnActive ? 10 : 20;

      this.camera.position.x = THREE.MathUtils.damp(this.camera.position.x, _desiredCamPos.x, posLambda, dt);
      this.camera.position.y = THREE.MathUtils.damp(this.camera.position.y, _desiredCamPos.y, posLambda, dt);
      this.camera.position.z = THREE.MathUtils.damp(this.camera.position.z, _desiredCamPos.z, posLambda, dt);

      _currentCamLook.x = THREE.MathUtils.damp(_currentCamLook.x, _desiredLook.x, lookLambda, dt);
      _currentCamLook.y = THREE.MathUtils.damp(_currentCamLook.y, _desiredLook.y, lookLambda, dt);
      _currentCamLook.z = THREE.MathUtils.damp(_currentCamLook.z, _desiredLook.z, lookLambda, dt);
    }

    if (this.cameraShake > 0.002) {
      this.cameraShake = THREE.MathUtils.damp(this.cameraShake, 0, 14, dt);
      this.camera.position.x += (Math.random() - 0.5) * this.cameraShake;
      this.camera.position.y += (Math.random() - 0.5) * this.cameraShake * 0.7;
    }

    const targetFov =
      thunderShotIdx === 0
        ? THREE.MathUtils.lerp(34, 30, Math.min(1, cutIn / 0.33))
        : thunderShotIdx === 1
          ? 27
          : thunderShotIdx === 2
            ? 36
            : thunderShotIdx === 3
              ? 50
              : isThunderLunge
                ? 58
                : isSniperAds
                  ? 20
                  : this.player.isDashing
                    ? 55
                    : 48;
    if (snapCut) {
      this.camera.fov = targetFov;
      this.camera.updateProjectionMatrix();
    } else if (Math.abs(this.camera.fov - targetFov) > 0.05) {
      this.camera.fov = THREE.MathUtils.damp(this.camera.fov, targetFov, cutIn >= 0 ? 12 : 13, dt);
      this.camera.updateProjectionMatrix();
    }

    this.camera.lookAt(_currentCamLook);
  }

  private getScopeTelemetry() {
    const targetDistance = Math.round(this.player.pos.distanceTo(this.enemy.pos) * 10) / 10;
    const camDir = _v1;
    this.camera.getWorldDirection(camDir);
    const aimPitchDeg = Math.round(Math.asin(THREE.MathUtils.clamp(camDir.y, -0.99, 0.99)) * (180 / Math.PI) * 10) / 10;

    const targetCenter = _v2.set(this.enemy.pos.x, this.enemy.pos.y + 1.02, this.enemy.pos.z);
    const rayToTarget = _v3.copy(targetCenter).sub(this.camera.position);
    const projAlongRay = rayToTarget.dot(camDir);
    const perpDist = projAlongRay > 0 ? rayToTarget.addScaledVector(camDir, -projAlongRay).length() : Infinity;
    const aimInfo = this.getPlayerAimInfo(this.enemy);
    const targetInReticle =
      this.enemy.hp > 0 && (aimInfo.hasSoftLock || (this.player.isAiming && perpDist < 1.15));

    const shotReadyPct =
      this.player.attackCooldown > 0
        ? Math.round(
            THREE.MathUtils.clamp(
              (1 - this.player.attackCooldown / Math.max(0.1, this.player.attackDuration)) * 100,
              0,
              100,
            ),
          )
        : 100;

    return {
      targetDistance,
      aimPitchDeg,
      targetInReticle,
      shotReadyPct,
    };
  }

  private emitHud() {
    const scopeTelemetry = this.getScopeTelemetry();
    const snapshot: BattleHudSnapshot = {
      player: {
        name: this.player.name,
        lbxName: this.player.loadout.name,
        hp: Math.ceil(this.player.hp),
        maxHp: this.player.stats.maxHp,
        energy: Math.round(this.player.energy),
        maxEnergy: this.player.stats.maxEnergy,
        special: Math.round(this.player.special),
        isGuarding: this.player.isGuarding,
        canGuard: this.player.stats.canGuard,
        isAiming: this.player.isAiming,
        isDashing: this.player.isDashing,
        isStunned: this.player.stunTimer > 0,
        comboCount: this.player.comboCounter,
      },
      enemy: {
        name: this.enemy.name,
        lbxName: this.enemy.loadout.name,
        hp: Math.ceil(this.enemy.hp),
        maxHp: this.enemy.stats.maxHp,
        energy: Math.round(this.enemy.energy),
        maxEnergy: this.enemy.stats.maxEnergy,
        special: Math.round(this.enemy.special),
        isGuarding: this.enemy.isGuarding,
        canGuard: this.enemy.stats.canGuard,
        isAiming: this.enemy.isAiming,
        isDashing: this.enemy.isDashing,
        isStunned: this.enemy.stunTimer > 0,
        comboCount: this.enemy.comboCounter,
        aiState: this.enemy.aiState,
      },
      elapsedTime: this.elapsedTime,
      lockOnActive: this.lockOnActive,
      bannerText: this.bannerTimer > 0 ? this.bannerText : null,
      specialCutIn:
        this.thunderCutInProgress >= 0
          ? {
              title: this.player.weapon.specialName.toUpperCase(),
              subtitle: `ЭКСТРА-ПРИЁМ • ${this.player.loadout.name.toUpperCase()}`,
              accentHex: this.player.weapon.accentHex,
              progress: this.thunderCutInProgress,
            }
          : null,
      outcome: this.outcome,
      targetDistance: scopeTelemetry.targetDistance,
      aimPitchDeg: scopeTelemetry.aimPitchDeg,
      targetInReticle: scopeTelemetry.targetInReticle,
      shotReadyPct: scopeTelemetry.shotReadyPct,
      stats: {
        damageDealt: this.damageDealt,
        parriesLanded: this.parriesLanded,
        specialsUsed: this.specialsUsed,
      },
    };
    this.callbacks.onHudUpdate(snapshot);
  }

  private afterDraw(cb: () => void) {
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const sync = gl.fenceSync?.(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!sync) {
      this.frameId = window.setTimeout(cb, 80);
      return;
    }
    gl.flush();
    const poll = () => {
      if (this.disposed) return;
      const state = gl.getSyncParameter(sync, gl.SYNC_STATUS);
      if (state === gl.UNSIGNALED) {
        this.frameId = window.setTimeout(poll, 5);
      } else {
        gl.deleteSync(sync);
        this.frameId = window.setTimeout(cb, 80);
      }
    };
    this.frameId = window.setTimeout(poll, 5);
  }

  private tick = () => {
    if (this.disposed) return;

    const rawDt = this.clock.getDelta();
    const dt = Math.min(rawDt, 0.05);

    if (!this.paused) {
      this.elapsedTime += dt;
      if (this.bannerTimer > 0) {
        this.bannerTimer = Math.max(0, this.bannerTimer - dt);
      }

      this.updateInteractables(dt);

      // Refresh active obstacles scratch array
      this.activeObstaclesScratch.length = 0;
      for (let i = 0; i < this.obstacleStates.length; i++) {
        if (!this.obstacleStates[i].destroyed) {
          this.activeObstaclesScratch.push(this.obstacleStates[i].data);
        }
      }

      if (this.outcome === 'ongoing') {
        const lanceProg =
          this.player.isSpecialAttack &&
          this.player.weapon.kind === 'lance' &&
          this.player.attackTimer > 0
            ? 1 - this.player.attackTimer / Math.max(0.1, this.player.attackDuration)
            : 1;
        const inCutIn =
          this.player.isSpecialAttack &&
          this.player.weapon.kind === 'lance' &&
          this.player.attackTimer > 0 &&
          lanceProg < 0.68;
        const enemyStandStill =
          this.player.isSpecialAttack &&
          this.player.weapon.kind === 'lance' &&
          this.player.attackTimer > 0 &&
          lanceProg < 0.88 &&
          !this.player.hasHitCurrentSwing;
        this.thunderCutInProgress = inCutIn
          ? THREE.MathUtils.clamp(lanceProg / 0.68, 0, 1)
          : -1;
        const worldDt = inCutIn ? dt * 0.03 : dt;

        const moveInput = inCutIn ? { x: 0, z: 0 } : this.getPlayerMoveInput();
        const wantAttack = inCutIn ? false : this.mouseAttackQueued;
        this.mouseAttackQueued = false;
        const wantGuard = inCutIn ? false : this.mouseGuardHeld || this.keys.has('KeyK');
        const pAim = this.getPlayerAimInfo(this.enemy).attackYaw;

        this.updateFighter(
          this.player,
          this.enemy,
          moveInput,
          wantAttack,
          wantGuard,
          false,
          false,
          pAim,
          dt,
        );

        const aiOut = this.aiBrain.update({
          aiPos: this.enemy.pos,
          aiHp: this.enemy.hp,
          aiMaxHp: this.enemy.stats.maxHp,
          aiEnergy: this.enemy.energy,
          aiMaxEnergy: this.enemy.stats.maxEnergy,
          aiSpecial: this.enemy.special,
          aiAttackCooldown: this.enemy.attackCooldown,
          aiComboStep: this.enemy.comboStep,
          aiIsStunned: this.enemy.stunTimer > 0,
          aiWeapon: this.enemy.weapon,
          aiCanGuard: this.enemy.stats.canGuard,
          aiIsGrounded: this.enemy.isGrounded,
          playerPos: this.player.pos,
          playerHp: this.player.hp,
          playerMaxHp: this.player.stats.maxHp,
          playerEnergy: this.player.energy,
          playerIsAttacking: this.player.attackPhase !== 'none',
          playerAttackPhase: this.player.attackPhase,
          playerIsGuarding: this.player.isGuarding,
          playerWeaponIsRanged: this.player.weapon.isRanged,
          playerFacingAngle: this.player.yaw,
          arenaRadius: this.arena.radius,
          obstacles: this.activeObstaclesScratch,
          readyInteractables: this.readyInteractablesScratch,
          dt: worldDt,
        });

        this.enemy.aiState = aiOut.tacticalState;
        if (enemyStandStill) {
          this.enemy.vel.x = 0;
          this.enemy.vel.z = 0;
          this.enemy.isDashing = false;
          this.enemy.isGuarding = false;
          this.enemy.isAiming = false;
          this.enemy.attackPhase = 'none';
          this.enemy.isSpecialAttack = false;
          stopCustomSpecialClip(this.enemy.rig);
        } else if (aiOut.wantJump && !inCutIn) {
          this.tryJump(this.enemy);
        }

        this.updateFighter(
          this.enemy,
          this.player,
          enemyStandStill ? { x: 0, z: 0 } : { x: aiOut.moveDirX, z: aiOut.moveDirZ },
          enemyStandStill || inCutIn ? false : aiOut.wantAttack,
          enemyStandStill || inCutIn ? false : aiOut.wantGuard,
          enemyStandStill || inCutIn ? false : aiOut.wantDash,
          enemyStandStill || inCutIn ? false : aiOut.wantSpecial,
          aiOut.aimAngle,
          worldDt,
        );

        this.updateProjectiles(worldDt);
      } else {
        this.thunderCutInProgress = -1;
        this.updateFighter(this.player, this.enemy, { x: 0, z: 0 }, false, false, false, false, null, dt * 0.65);
        this.updateFighter(this.enemy, this.player, { x: 0, z: 0 }, false, false, false, false, null, dt * 0.65);
        if (this.outcomeDelayTimer > 0) {
          this.outcomeDelayTimer -= dt;
          if (this.outcomeDelayTimer <= 0) {
            this.emitHud();
            const finalSnap: BattleHudSnapshot = {
              player: {
                name: this.player.name,
                lbxName: this.player.loadout.name,
                hp: Math.ceil(this.player.hp),
                maxHp: this.player.stats.maxHp,
                energy: Math.round(this.player.energy),
                maxEnergy: this.player.stats.maxEnergy,
                special: Math.round(this.player.special),
                isGuarding: false,
                canGuard: this.player.stats.canGuard,
                isAiming: false,
                isDashing: false,
                isStunned: false,
                comboCount: 0,
              },
              enemy: {
                name: this.enemy.name,
                lbxName: this.enemy.loadout.name,
                hp: Math.ceil(this.enemy.hp),
                maxHp: this.enemy.stats.maxHp,
                energy: Math.round(this.enemy.energy),
                maxEnergy: this.enemy.stats.maxEnergy,
                special: Math.round(this.enemy.special),
                isGuarding: false,
                canGuard: this.enemy.stats.canGuard,
                isAiming: false,
                isDashing: false,
                isStunned: false,
                comboCount: 0,
                aiState: this.enemy.aiState,
              },
              elapsedTime: this.elapsedTime,
              lockOnActive: this.lockOnActive,
              bannerText: null,
              specialCutIn: null,
              outcome: this.outcome,
              stats: {
                damageDealt: this.damageDealt,
                parriesLanded: this.parriesLanded,
                specialsUsed: this.specialsUsed,
              },
            };
            this.callbacks.onBattleEnd(this.outcome, finalSnap);
          }
        }
        this.updateProjectiles(dt);
      }

      this.updateVfx(dt);
      this.updateCamera(dt);

      if (this.sunLight) {
        // Snap shadow frustum center to shadow texel grid around the player for crisp anti-shimmer sunlight across the 54m valley
        const texelSnap = (60 / 2048) * 8;
        const focusX = Math.round(this.player.pos.x / texelSnap) * texelSnap;
        const focusZ = Math.round(this.player.pos.z / texelSnap) * texelSnap;
        this.sunLight.target.position.set(focusX, 0, focusZ);
        this.sunLight.position
          .copy(this.sunLight.target.position)
          .addScaledVector(this.sunDir, 62);
        this.sunLight.target.updateMatrixWorld();
      }

      this.hudEmitTimer += dt;
      if (this.hudEmitTimer >= 0.05) {
        this.hudEmitTimer = 0;
        this.emitHud();
      }
    }

    this.renderer.render(this.scene, this.camera);

    if (this.isSoftwareGl) {
      this.afterDraw(this.tick);
    } else {
      this.frameId = requestAnimationFrame(this.tick);
    }
  };

  public dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    clearTimeout(this.frameId);

    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('resize', this.onResize);
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    this.exitPointerLock();
    const dom = this.renderer.domElement;
    dom.removeEventListener('mousedown', this.onMouseDown);
    window.removeEventListener('mouseup', this.onMouseUp);
    dom.removeEventListener('mouseenter', this.onMouseEnter);
    dom.removeEventListener('mouseleave', this.onMouseLeave);
    dom.removeEventListener('mousemove', this.onMouseMove);
    dom.removeEventListener('wheel', this.onWheel);
    dom.removeEventListener('touchstart', this.onTouchStart);
    dom.removeEventListener('touchmove', this.onTouchMove);
    dom.removeEventListener('touchend', this.onTouchEnd);
    dom.removeEventListener('contextmenu', this.onContextMenu);

    this.player.rig.dispose();
    this.enemy.rig.dispose();
    this.sharedGeos.forEach((g) => g.dispose());
    this.sharedMats.forEach((m) => m.dispose());
    this.disposeArena();
    this.disposeEnv();
    this.renderer.dispose();
    if (dom.parentElement === this.container) {
      this.container.removeChild(dom);
    }
  }
}
